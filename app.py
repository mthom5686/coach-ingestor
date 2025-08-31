# app.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime
from decimal import Decimal
import os, io, csv, json, re
import httpx
from bs4 import BeautifulSoup

from sqlalchemy import (
    create_engine, Column, Integer, String, Date, ForeignKey, JSON,
    UniqueConstraint, BigInteger, Numeric, text
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from sqlalchemy.exc import IntegrityError

# =========================
# DB setup
# =========================
DB_HOST = os.getenv("DB_HOST", "coachdb")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "coach")
DB_USER = os.getenv("DB_USER", "coach")
DB_PASS = os.getenv("DB_PASS", "coachpass")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# =========================
# Models
# =========================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    protein_target = Column(Integer, nullable=True)

    # new goal fields (added at runtime via ALTER TABLE IF NOT EXISTS)
    # stored as columns but defined here for ORM usage
    # (metadata.create_all doesn't add columns to existing table, so we DDL them at startup)
    # types match the ALTER TABLE below:
    # goal_weight_kg: Numeric(6,2)   goal_date: Date
    goal_weight_kg = Column(Numeric(6, 2), nullable=True)
    goal_date = Column(Date, nullable=True)

    nutrition_provider = Column(String, nullable=True)
    nutrition_api_base = Column(String, nullable=True)
    nutrition_api_key = Column(String, nullable=True)

    workouts = relationship("HevyWorkout", back_populates="user", cascade="all, delete-orphan")
    nutrition_logs = relationship("NutritionLog", back_populates="user", cascade="all, delete-orphan")
    weight_logs = relationship("WeightLog", back_populates="user", cascade="all, delete-orphan")

class HevyWorkout(Base):
    __tablename__ = "hevy_workouts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    date = Column(Date, index=True, nullable=False)
    duration_minutes = Column(Integer, nullable=True)
    total_volume = Column(BigInteger, nullable=True)
    source_url = Column(String, nullable=False)
    raw = Column(JSON, nullable=True)
    user = relationship("User", back_populates="workouts")
    __table_args__ = (UniqueConstraint("user_id", "source_url", name="uq_workout_user_source"),)

class NutritionLog(Base):
    __tablename__ = "nutrition_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    date = Column(Date, index=True, nullable=False)
    protein_g = Column(Integer, nullable=True)
    calories = Column(Integer, nullable=True)
    carbs_g = Column(Integer, nullable=True)
    fat_g = Column(Integer, nullable=True)
    source = Column(String, nullable=True)
    user = relationship("User", back_populates="nutrition_logs")
    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_nutrition_user_date"),)

class WeightLog(Base):
    __tablename__ = "weight_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    date = Column(Date, index=True, nullable=False)
    weight_kg = Column(Numeric(6, 2), nullable=False)
    source = Column(String, nullable=True)
    user = relationship("User", back_populates="weight_logs")
    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_weight_user_date"),)

# Create tables that don't exist
Base.metadata.create_all(bind=engine)

# Add new columns to users table if missing
def init_schema():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS goal_weight_kg numeric(6,2)"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS goal_date date"))

init_schema()

# =========================
# Schemas
# =========================
class UserCreate(BaseModel):
    name: str = Field(..., min_length=2)
    protein_target: Optional[int] = None
    nutrition_provider: Optional[str] = None
    nutrition_api_base: Optional[str] = None
    nutrition_api_key: Optional[str] = None

class GoalIn(BaseModel):
    user_name: str
    goal_weight_kg: float
    goal_date: date

class UserOut(BaseModel):
    id: int
    name: str
    protein_target: Optional[int] = None
    goal_weight_kg: Optional[float] = None
    goal_date: Optional[date] = None
    class Config:
        from_attributes = True

class HevyIn(BaseModel):
    url: str
    user_name: str

class HevyWorkoutOut(BaseModel):
    user_name: str
    date: date
    duration_minutes: Optional[int]
    total_volume: Optional[int]

class NutritionManualIn(BaseModel):
    user_name: str
    date: date
    protein_g: int
    calories: Optional[int] = None
    carbs_g: Optional[int] = None
    fat_g: Optional[int] = None

class WeightManualIn(BaseModel):
    user_name: str
    date: date
    weight_kg: float

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Coach Ingestor API (single-file)", version="0.2.0")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

# =========================
# CRUD helpers
# =========================
def get_or_create_user(db: Session, name: str, **kwargs) -> User:
    u = db.query(User).filter(User.name == name).first()
    if u:
        for k, v in kwargs.items():
            if v is not None and hasattr(u, k):
                setattr(u, k, v)
        db.commit()
        db.refresh(u)
        return u
    u = User(name=name, **kwargs)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u

def create_workout(db: Session, user_id: int, dt: date, duration: int|None, volume: int|None, url: str, raw: dict|None):
    safe_raw = None
    if raw is not None:
        safe_raw = jsonable_encoder(
            raw,
            custom_encoder={
                date: lambda v: v.isoformat(),
                datetime: lambda v: v.isoformat(),
            },
        )
    w = HevyWorkout(
        user_id=user_id,
        date=dt,
        duration_minutes=duration,
        total_volume=volume,
        source_url=url,
        raw=safe_raw,
    )
    db.add(w)
    db.commit()
    db.refresh(w)
    return w

def upsert_nutrition(db: Session, user_id: int, dt: date, protein_g: int|None, calories: int|None, carbs_g: int|None, fat_g: int|None, source: str="api"):
    log = db.query(NutritionLog).filter(NutritionLog.user_id==user_id, NutritionLog.date==dt).first()
    if not log:
        log = NutritionLog(user_id=user_id, date=dt); db.add(log)
    log.protein_g, log.calories, log.carbs_g, log.fat_g, log.source = protein_g, calories, carbs_g, fat_g, source
    db.commit(); db.refresh(log); return log

def upsert_weight(db: Session, user_id: int, dt: date, weight_kg: float|Decimal, source: str="manual"):
    log = db.query(WeightLog).filter(WeightLog.user_id==user_id, WeightLog.date==dt).first()
    if not log:
        log = WeightLog(user_id=user_id, date=dt, weight_kg=Decimal(str(weight_kg)), source=source)
        db.add(log)
    else:
        log.weight_kg = Decimal(str(weight_kg))
        log.source = source
    db.commit(); db.refresh(log); return log

# =========================
# HTTP fetch headers (browser-like)
# =========================
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

# =========================
# Hevy parser (robust)
# =========================
def extract_next_data(html: str) -> dict | None:
    m = re.search(r'<script[^>]+id=[\'"]__NEXT_DATA__[\'"][^>]*>(.*?)</script>', html, re.S | re.I)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def walk_obj(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_obj(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from walk_obj(it)

def _num_from_any(x):
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, dict):
        for k in ("kg", "value", "val", "amount", "weight"):
            if k in x: return _num_from_any(x[k])
        return None
    if isinstance(x, str):
        m = re.search(r'[-+]?\d*\.?\d+', x)
        return float(m.group(0)) if m else None
    return None

def _weight_kg_from_set(s: dict):
    candidates = [
        s.get("weight"), s.get("w"), s.get("weightKg"), s.get("weight_kg"),
        s.get("kg"), s.get("weightValue"), s.get("value"), s.get("mass"),
        s.get("load"), s.get("weightLbs"), s.get("lbs"), s.get("lb"),
    ]
    unit = (s.get("unit") or s.get("weightUnit") or s.get("u") or "").lower()
    raw = next((c for c in candidates if c not in (None, "")), None)
    val = _num_from_any(raw)
    txt = str(raw).lower()
    looks_lb = ("lbs" in txt or " lb" in txt or unit in ("lb", "lbs", "pounds"))
    looks_kg = ("kg" in txt) or (unit in ("kg", "kilogram", "kilograms"))
    if val is None: return 0.0
    if looks_lb and not looks_kg: return round(val * 0.45359237, 3)
    return float(val)

def parse_duration_minutes(val, key_hint: str = "") -> int | None:
    if val is None: return None
    if isinstance(val, dict):
        for kk in ("minutes","mins","min"):
            if kk in val and val[kk] is not None:
                try: return int(round(float(val[kk])))
                except: pass
        for kk in ("seconds","secs","sec"):
            if kk in val and val[kk] is not None:
                try: return int(round(float(val[kk]) / 60.0))
                except: pass
        unit = str(val.get("unit", "")).lower()
        if "value" in val and val["value"] is not None:
            try:
                v = float(val["value"])
                if unit in ("s","sec","secs","second","seconds"): return int(round(v/60.0))
                if unit in ("m","min","mins","minute","minutes"): return int(round(v))
            except: pass
        for kk in ("value","val","amount","duration"):
            if kk in val and val[kk] is not None:
                return parse_duration_minutes(val[kk], key_hint)
        return None
    if isinstance(val, (int, float)):
        v = float(val); kh = key_hint.lower()
        if any(x in kh for x in ("sec","second")): return int(round(v/60.0))
        if any(x in kh for x in ("min","minute")): return int(round(v))
        if v > 600: return int(round(v/60.0))
        return int(round(v))
    if isinstance(val, str):
        s = val.strip().lower()
        m = re.match(r"^pt(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", s)
        if m:
            h = int(m.group(1) or 0); mi = int(m.group(2) or 0); se = int(m.group(3) or 0)
            return h*60 + mi + int(round(se/60.0))
        m = re.match(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})$", s)
        if m:
            h = int(m.group(1) or 0); mi = int(m.group(2) or 0); se = int(m.group(3) or 0)
            return h*60 + mi + int(round(se/60.0))
        h = re.search(r"(\d+)\s*h", s)
        mi = re.search(r"(\d+)\s*m(?:in(?:ute)?s?)?\b", s)
        se = re.search(r"(\d+)\s*s(?:ec(?:ond)?s?)?\b", s)
        if h or mi or se:
            total = 0
            if h: total += int(h.group(1))*60
            if mi: total += int(mi.group(1))
            if se: total += int(round(int(se.group(1))/60.0))
            if total > 0: return total
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(min|mins|minute|minutes|m)\b", s)
        if m: return int(round(float(m.group(1))))
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(sec|secs|second|seconds|s)\b", s)
        if m: return int(round(float(m.group(1))/60.0))
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*$", s)
        if m:
            v = float(m.group(1)); return parse_duration_minutes(v, key_hint)
    return None

def parse_hevy_share(url: str) -> dict:
    with httpx.Client(
        timeout=httpx.Timeout(8.0, connect=5.0, read=8.0),
        follow_redirects=True,
        headers={**HTTP_HEADERS, "Referer": "https://hevy.com/"},
        transport=httpx.HTTPTransport(retries=2),
    ) as client:
        r = client.get(url)
        r.raise_for_status()
        html = r.text

    soup = BeautifulSoup(html, "html.parser")

    data_sources = []
    nd = extract_next_data(html)
    if nd: data_sources.append(nd)

    for blob in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.S | re.I):
        try: data_sources.append(json.loads(blob))
        except: pass

    exercises = []
    duration_minutes = None
    wdate = None

    for root in data_sources:
        for node in walk_obj(root):
            ex = node.get("exercises") or node.get("workoutExercises") or node.get("setsByExercise")
            if isinstance(ex, list) and ex:
                for e in ex:
                    name = e.get("name") or e.get("exerciseName") or "Unknown"
                    raw_sets = e.get("sets") or e.get("workoutSets") or []
                    parsed_sets = []
                    for s in raw_sets:
                        reps = s.get("reps") or s.get("repetitions") or s.get("r") or 0
                        reps = int(_num_from_any(reps) or 0)
                        wkg = _weight_kg_from_set(s)
                        parsed_sets.append({"reps": reps, "weight_kg": wkg})
                    exercises.append({"name": name, "sets": parsed_sets})

            # scan broadly for any duration/time/elapsed key
            if duration_minutes is None and isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    if any(key in kl for key in ("duration", "time", "elapsed")):
                        mins = parse_duration_minutes(v, key_hint=kl)
                        if mins is not None and 0 < mins < 600:  # sanity: ignore absurd values
                            duration_minutes = mins
                            break

            if not wdate:
                wdate = node.get("date") or node.get("startTime") or node.get("startDate") or node.get("workoutDate")

    if duration_minutes is None:
        html_lower = html.lower()
        m = re.search(r"pt(?:\d+h)?(?:\d+m)?(?:\d+s)?", html_lower)
        if m:
            guess = parse_duration_minutes(m.group(0))
            if guess and 0 < guess < 600:
                duration_minutes = guess
    if duration_minutes is None:
        m = re.search(r"(\d+)\s*h(?:\s*(\d+)\s*m)?", html_lower) or re.search(r"(\d+)\s*m(?:in(?:ute)?s?)?\b", html_lower)
        if m:
            guess = parse_duration_minutes(m.group(0))
            if guess and 0 < guess < 600:
                duration_minutes = guess

    total_volume = 0
    for e in exercises:
        for s in e["sets"]:
            total_volume += int(round((s.get("weight_kg") or 0) * (s.get("reps") or 0)))

    parsed_date = None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed_date = datetime.strptime(str(wdate)[:len(fmt)], fmt).date()
            break
        except:
            continue

    return {
        "date": parsed_date or datetime.utcnow().date(),
        "duration_minutes": int(duration_minutes) if duration_minutes is not None else None,
        "total_volume": int(total_volume),
        "exercises": exercises,
        "raw_hint": "Parsed from Hevy share (weights normalized to kg)",
    }

# =========================
# Routes
# =========================
@app.post("/users", response_model=UserOut)
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    return get_or_create_user(
        db,
        name=body.name,
        protein_target=body.protein_target,
        nutrition_provider=body.nutrition_provider,
        nutrition_api_base=body.nutrition_api_base,
        nutrition_api_key=body.nutrition_api_key,
    )

@app.post("/users/goal")
def set_goal(body: GoalIn, db: Session = Depends(get_db)):
    u = get_or_create_user(db, name=body.user_name)
    u.goal_weight_kg = Decimal(str(body.goal_weight_kg))
    u.goal_date = body.goal_date
    db.commit(); db.refresh(u)
    return {"ok": True, "user": u.name, "goal_weight_kg": float(u.goal_weight_kg), "goal_date": u.goal_date.isoformat()}

@app.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.post("/ingest/hevy", response_model=HevyWorkoutOut)
def ingest_hevy(body: HevyIn, db: Session = Depends(get_db)):
    user = get_or_create_user(db, name=body.user_name)
    try:
        parsed = parse_hevy_share(body.url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Hevy fetch/parse failed: {type(e).__name__}: {e}")

    safe_raw = jsonable_encoder(parsed, custom_encoder={
        date: lambda v: v.isoformat(),
        datetime: lambda v: v.isoformat(),
    })

    existing = db.query(HevyWorkout).filter(
        HevyWorkout.user_id == user.id,
        HevyWorkout.source_url == body.url
    ).first()

    if existing:
        existing.date = parsed["date"]
        existing.duration_minutes = parsed.get("duration_minutes")
        existing.total_volume = parsed.get("total_volume")
        existing.raw = safe_raw
        db.commit(); db.refresh(existing)
        w = existing
    else:
        try:
            w = HevyWorkout(
                user_id=user.id,
                date=parsed["date"],
                duration_minutes=parsed.get("duration_minutes"),
                total_volume=parsed.get("total_volume"),
                source_url=body.url,
                raw=safe_raw,
            )
            db.add(w); db.commit(); db.refresh(w)
        except IntegrityError:
            db.rollback()
            existing = db.query(HevyWorkout).filter(
                HevyWorkout.user_id == user.id,
                HevyWorkout.source_url == body.url
            ).first()
            if not existing:
                raise HTTPException(status_code=409, detail="Workout exists and could not be updated")
            existing.date = parsed["date"]
            existing.duration_minutes = parsed.get("duration_minutes")
            existing.total_volume = parsed.get("total_volume")
            existing.raw = safe_raw
            db.commit(); db.refresh(existing)
            w = existing

    return HevyWorkoutOut(
        user_name=user.name,
        date=w.date,
        duration_minutes=w.duration_minutes,
        total_volume=w.total_volume,
    )

@app.post("/ingest/nutrition/manual")
def ingest_nutrition_manual(body: NutritionManualIn, db: Session = Depends(get_db)):
    user = get_or_create_user(db, name=body.user_name)
    upsert_nutrition(
        db, user_id=user.id, dt=body.date,
        protein_g=body.protein_g, calories=body.calories,
        carbs_g=body.carbs_g, fat_g=body.fat_g, source="manual"
    )
    return {"ok": True}

@app.post("/ingest/nutrition/csv")
def ingest_nutrition_csv(user_name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    user = get_or_create_user(db, name=user_name)
    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    n = 0
    for row in reader:
        dt = date.fromisoformat(row["date"])
        protein = int(float(row.get("protein_g", 0)))
        calories = int(float(row.get("calories", 0))) if row.get("calories") else None
        carbs = int(float(row.get("carbs_g", 0))) if row.get("carbs_g") else None
        fat = int(float(row.get("fat_g", 0))) if row.get("fat_g") else None
        upsert_nutrition(
            db, user_id=user.id, dt=dt, protein_g=protein,
            calories=calories, carbs_g=carbs, fat_g=fat, source="csv"
        )
        n += 1
    return {"ok": True, "rows": n}

@app.post("/ingest/weight/manual")
def ingest_weight_manual(body: WeightManualIn, db: Session = Depends(get_db)):
    user = get_or_create_user(db, name=body.user_name)
    upsert_weight(db, user_id=user.id, dt=body.date, weight_kg=body.weight_kg, source="manual")
    return {"ok": True}

@app.post("/ingest/weight/csv")
def ingest_weight_csv(user_name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    user = get_or_create_user(db, name=user_name)
    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    n = 0
    for row in reader:
        dt = date.fromisoformat(row["date"])
        wkg = float(row["weight_kg"])
        upsert_weight(db, user_id=user.id, dt=dt, weight_kg=wkg, source="csv")
        n += 1
    return {"ok": True, "rows": n}

# --- debug: recent workouts
@app.get("/debug/workouts")
def debug_workouts(user_name: Optional[str] = None, limit: int = 10, db: Session = Depends(get_db)):
    q = db.query(HevyWorkout, User).join(User, HevyWorkout.user_id == User.id)
    if user_name:
        q = q.filter(User.name == user_name)
    rows = q.order_by(HevyWorkout.id.desc()).limit(limit).all()
    return [{"user": u.name, "date": w.date.isoformat(), "volume": w.total_volume, "url": w.source_url} for (w, u) in rows]

# --- debug: parse a hevy link without inserting
@app.get("/debug/hevy")
def debug_hevy(url: str):
    try:
        parsed = parse_hevy_share(url)
        return {
            "ok": True,
            "date": parsed.get("date").isoformat() if parsed.get("date") else None,
            "duration_minutes": parsed.get("duration_minutes"),
            "total_volume": parsed.get("total_volume"),
            "exercise_count": len(parsed.get("exercises") or []),
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# --- debug: nutrition rows
@app.get("/debug/nutrition")
def debug_nutrition(user_name: Optional[str] = None, limit: int = 20, db: Session = Depends(get_db)):
    q = db.query(NutritionLog, User).join(User, NutritionLog.user_id == User.id)
    if user_name:
        q = q.filter(User.name == user_name)
    rows = q.order_by(NutritionLog.id.desc()).limit(limit).all()
    return [
        {"user": u.name, "date": n.date.isoformat(), "protein_g": n.protein_g}
        for (n, u) in rows
    ]

# --- debug: weight rows
@app.get("/debug/weight")
def debug_weight(user_name: Optional[str] = None, limit: int = 30, db: Session = Depends(get_db)):
    q = db.query(WeightLog, User).join(User, WeightLog.user_id == User.id)
    if user_name:
        q = q.filter(User.name == user_name)
    rows = q.order_by(WeightLog.date.desc()).limit(limit).all()
    return [
        {"user": u.name, "date": w.date.isoformat(), "weight_kg": float(w.weight_kg)}
        for (w, u) in rows
    ]

# --- debug: pace calc (simple)
@app.get("/debug/pace")
def debug_pace(user_name: str, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.name == user_name).first()
    if not u:
        raise HTTPException(404, "user not found")
    if not u.goal_date or not u.goal_weight_kg:
        raise HTTPException(400, "set goal first via POST /users/goal")

    # first and last weight
    first = db.query(WeightLog).filter(WeightLog.user_id==u.id).order_by(WeightLog.date.asc()).first()
    last  = db.query(WeightLog).filter(WeightLog.user_id==u.id).order_by(WeightLog.date.desc()).first()
    if not first or not last:
        raise HTTPException(400, "no weight logs")

    start_w, start_d = float(first.weight_kg), first.date
    last_w, last_d   = float(last.weight_kg), last.date
    goal_w, goal_d   = float(u.goal_weight_kg), u.goal_date

    total_days = max(1, (goal_d - start_d).days)
    elapsed = (last_d - start_d).days
    progress_ratio = (last_w - start_w) / (goal_w - start_w) if (goal_w - start_w) != 0 else 0.0
    scheduled_days_by_now = progress_ratio * total_days
    days_ahead = round(scheduled_days_by_now - elapsed, 1)

    # scheduled weight for today
    today = date.today()
    t_elapsed = max(0, min(total_days, (today - start_d).days))
    scheduled_today = start_w + (t_elapsed / total_days) * (goal_w - start_w)

    return {
        "user": u.name,
        "start": {"date": start_d.isoformat(), "weight_kg": start_w},
        "latest": {"date": last_d.isoformat(), "weight_kg": last_w},
        "goal": {"date": goal_d.isoformat(), "weight_kg": goal_w},
        "scheduled_today": round(scheduled_today, 2),
        "days_ahead": days_ahead
    }

# --- debug: raw fetch tester
@app.get("/debug/fetch")
def debug_fetch(url: str):
    try:
        with httpx.Client(
            timeout=httpx.Timeout(8.0, connect=5.0, read=8.0),
            follow_redirects=True,
            headers={**HTTP_HEADERS, "Referer": "https://hevy.com/"},
            transport=httpx.HTTPTransport(retries=2),
        ) as client:
            r = client.get(url)
            return {"ok": True, "status": r.status_code, "final_url": str(r.url), "length": len(r.text)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# =========================
# Simple homepage
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>LiftCrew – Submit</title>
    <link rel="stylesheet" href="https://unpkg.com/@picocss/pico@2/css/pico.min.css">
    <style>
      .container { max-width: 880px; margin: 2rem auto; }
      .card { padding: 1.25rem; border: 1px solid #e7e7e9; border-radius: 12px; }
      .grid { display: grid; gap: 1rem; grid-template-columns: 1fr; }
      @media (min-width: 860px){ .grid { grid-template-columns: 1fr 1fr; } }
      .toast { position: fixed; top: 14px; right: 14px; background: #0ea5e9; color: white; padding: .75rem 1rem; border-radius: 10px; display:none; }
      .toast.error { background:#ef4444; }
      .muted { color: #6b7280; }
      .brand { font-weight: 700; font-size: 1.2rem; }
      footer { margin-top: 2rem; color:#6b7280; font-size:.9rem; }
      .kbd { background:#f1f5f9; border-radius:8px; padding:.15rem .4rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    </style>
  </head>
  <body>
    <div class="toast" id="toast"></div>
    <header class="container">
      <nav>
        <ul>
          <li class="brand">🏋️ LiftCrew</li>
        </ul>
        <ul>
          <li><a href="/docs" target="_blank">API</a></li>
          <li><a href="#dashboard" id="dashLink">Dashboard</a></li>
        </ul>
      </nav>
    </header>

    <main class="container">
      <h2>Log your stuff, win the week 💪</h2>
      <p class="muted">Paste your Hevy share link, log your bodyweight, or today’s protein.</p>

      <div class="grid">
        <section class="card">
          <h3>Share a Hevy workout</h3>
          <form id="hevyForm">
            <label for="hevyUser">Who are you?</label>
            <select id="hevyUser" required></select>

            <label for="hevyUrl">Hevy share link</label>
            <input id="hevyUrl" type="url" placeholder="https://hevy.app/workout/ABC123" required />

            <button type="submit">Submit workout</button>
            <small class="muted">Tip: in Hevy tap <span class="kbd">Share</span> → <span class="kbd">Copy link</span>.</small>
          </form>
        </section>

        <section class="card">
          <h3>Log bodyweight</h3>
          <form id="weightForm">
            <label for="wUser">Who are you?</label>
            <select id="wUser" required></select>

            <label for="wDate">Date</label>
            <input id="wDate" type="date" required />

            <label for="wKg">Weight (kg)</label>
            <input id="wKg" type="number" min="20" max="400" step="0.1" placeholder="e.g. 82.4" required />

            <button type="submit">Save weight</button>
          </form>
        </section>

        <section class="card">
          <h3>Log protein for today</h3>
          <form id="protForm">
            <label for="protUser">Who are you?</label>
            <select id="protUser" required></select>

            <label for="protDate">Date</label>
            <input id="protDate" type="date" required />

            <label for="protG">Protein (g)</label>
            <input id="protG" type="number" min="0" step="1" placeholder="e.g. 180" required />

            <button type="submit">Save protein</button>
            <small class="muted">Bulk import CSVs in <a href="/docs" target="_blank">API docs</a>.</small>
          </form>
        </section>
      </div>

      <section id="dashboard" style="margin-top:2rem;">
        <h3>Team dashboard</h3>
        <p class="muted">Open the shared Grafana board:</p>
        <a id="grafanaHref" class="secondary" target="_blank" href="#">Open Grafana</a>
      </section>

      <footer>
        <p>Pro tip: save this page to your phone’s home screen for 1-tap logging.</p>
      </footer>
    </main>

    <script>
    const API = window.location.origin;
    const toast = document.getElementById('toast');
    function showToast(msg, isErr=false){
      toast.textContent = msg; toast.className = 'toast' + (isErr?' error':'');
      toast.style.display = 'block'; setTimeout(()=>toast.style.display='none', 2600);
    }

    async function loadUsers() {
      try {
        const res = await fetch(API + '/users');
        const users = await res.json();
        for (const id of ['hevyUser','protUser','wUser']) {
          const sel = document.getElementById(id);
          sel.innerHTML = '';
          users.forEach(u => {
            const opt = document.createElement('option');
            opt.value = u.name; opt.textContent = u.name;
            sel.appendChild(opt);
          });
        }
      } catch (e) { showToast('Could not load users', true); }
    }

    // Set your Grafana link (replace with your real /d/<UID>/<slug>)
    document.getElementById('grafanaHref').href =
      'http://100.89.255.16:3030/d/f2da7bd4-0c35-48d5-9185-e31f546d709d/fitness-goal-tracker?orgId=1&var-user=All&from=now-30d&to=now&kiosk';
    document.getElementById('dashLink').href = document.getElementById('grafanaHref').href;
    document.getElementById('dashLink').setAttribute('target','_blank');

    // Hevy submit
    document.getElementById('hevyForm').addEventListener('submit', async (e)=>{
      e.preventDefault();
      const user = document.getElementById('hevyUser').value;
      const url = document.getElementById('hevyUrl').value.trim();
      try {
        const res = await fetch(API + '/ingest/hevy', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({user_name: user, url})
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        showToast(`Saved: ${data.total_volume?.toLocaleString()} total volume`);
        document.getElementById('hevyUrl').value = '';
      } catch(err){ showToast('Submit failed', true); }
    });

    // Weight submit
    document.getElementById('weightForm').addEventListener('submit', async (e)=>{
      e.preventDefault();
      const user = document.getElementById('wUser').value;
      const d = document.getElementById('wDate').value;
      const w = parseFloat(document.getElementById('wKg').value);
      try {
        const res = await fetch(API + '/ingest/weight/manual', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({user_name: user, date: d, weight_kg: w})
        });
        if (!res.ok) throw new Error(await res.text());
        showToast('Weight saved!');
        document.getElementById('wKg').value = '';
      } catch(err){ showToast('Save failed', true); }
    });

    // Protein submit
    document.getElementById('protForm').addEventListener('submit', async (e)=>{
      e.preventDefault();
      const user = document.getElementById('protUser').value;
      const d = document.getElementById('protDate').value;
      const g = parseInt(document.getElementById('protG').value || '0', 10);
      try {
        const res = await fetch(API + '/ingest/nutrition/manual', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({user_name: user, date: d, protein_g: g})
        });
        if (!res.ok) throw new Error(await res.text());
        showToast('Protein saved!');
        document.getElementById('protG').value = '';
      } catch(err){ showToast('Save failed', true); }
    });

    document.getElementById('protDate').valueAsDate = new Date();
    document.getElementById('wDate').valueAsDate = new Date();
    loadUsers();
    </script>
  </body>
</html>
    """
