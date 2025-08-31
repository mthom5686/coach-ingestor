from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime
import os, io, csv, json, re
import httpx
from bs4 import BeautifulSoup

from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey, JSON, UniqueConstraint, BigInteger
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# ----- DB setup -----
DB_HOST = os.getenv("DB_HOST", "coachdb")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "coach")
DB_USER = os.getenv("DB_USER", "coach")
DB_PASS = os.getenv("DB_PASS", "coachpass")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ----- Models -----
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    protein_target = Column(Integer, nullable=True)
    nutrition_provider = Column(String, nullable=True)
    nutrition_api_base = Column(String, nullable=True)
    nutrition_api_key = Column(String, nullable=True)
    workouts = relationship("HevyWorkout", back_populates="user", cascade="all, delete-orphan")
    nutrition_logs = relationship("NutritionLog", back_populates="user", cascade="all, delete-orphan")

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

Base.metadata.create_all(bind=engine)

# ----- Schemas -----
class UserCreate(BaseModel):
    name: str = Field(..., min_length=2)
    protein_target: Optional[int] = None
    nutrition_provider: Optional[str] = None
    nutrition_api_base: Optional[str] = None
    nutrition_api_key: Optional[str] = None

class UserOut(BaseModel):
    id: int
    name: str
    protein_target: Optional[int] = None
    class Config: from_attributes = True

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

# ----- FastAPI -----
app = FastAPI(title="Coach Ingestor API (single-file)", version="0.1.0")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@app.get("/health")
def health(): return {"ok": True, "ts": datetime.utcnow().isoformat()}

# ----- CRUD helpers -----
def get_or_create_user(db: Session, name: str, **kwargs) -> User:
    u = db.query(User).filter(User.name == name).first()
    if u:
        for k, v in kwargs.items():
            if v is not None and hasattr(u, k): setattr(u, k, v)
        db.commit(); db.refresh(u); return u
    u = User(name=name, **kwargs); db.add(u); db.commit(); db.refresh(u); return u

def create_workout(db: Session, user_id: int, dt: date, duration: int|None, volume: int|None, url: str, raw: dict|None):
    w = HevyWorkout(user_id=user_id, date=dt, duration_minutes=duration, total_volume=volume, source_url=url, raw=raw)
    db.add(w); db.commit(); db.refresh(w); return w

def upsert_nutrition(db: Session, user_id: int, dt: date, protein_g: int|None, calories: int|None, carbs_g: int|None, fat_g: int|None, source: str="api"):
    log = db.query(NutritionLog).filter(NutritionLog.user_id==user_id, NutritionLog.date==dt).first()
    if not log:
        log = NutritionLog(user_id=user_id, date=dt); db.add(log)
    log.protein_g, log.calories, log.carbs_g, log.fat_g, log.source = protein_g, calories, carbs_g, fat_g, source
    db.commit(); db.refresh(log); return log

# ----- Hevy parser (heuristic) -----
UA = {"User-Agent": "Mozilla/5.0 (CoachIngestor/1.0)"}

def extract_json_blobs(html: str):
    blobs = []
    m = re.search(r'__NEXT_DATA__"\s*type="application/json">(.+?)</script>', html)
    if m:
        try: blobs.append(json.loads(m.group(1)))
        except: pass
    for blob in re.findall(r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', html, flags=re.S):
        try: blobs.append(json.loads(blob))
        except: pass
    m3 = re.search(r'"exercises"\s*:\s*(\[[\s\S]*?\])', html)
    if m3:
        try: blobs.append({"exercises": json.loads(m3.group(1))})
        except: pass
    return blobs

def walk(obj, key_pred):
    if isinstance(obj, dict):
        if key_pred(obj.keys()): yield obj
        for v in obj.values(): yield from walk(v, key_pred)
    elif isinstance(obj, list):
        for it in obj: yield from walk(it, key_pred)

def parse_hevy_share(url: str) -> dict:
    with httpx.Client(timeout=20.0, headers=UA, follow_redirects=True) as client:
        r = client.get(url); r.raise_for_status(); html = r.text
    soup = BeautifulSoup(html, "lxml")
    collected = extract_json_blobs(html)

    exercises, duration_minutes, wdate = [], None, None
    for blob in collected:
        for n in walk(blob, lambda k: "exercises" in k):
            ex = n.get("exercises"); 
            if isinstance(ex, list) and ex:
                for e in ex:
                    name = e.get("name") or e.get("exerciseName") or "Unknown"
                    sets, raw_sets = [], (e.get("sets") or e.get("workoutSets") or [])
                    for s in raw_sets:
                        reps = s.get("reps") or s.get("repetitions") or s.get("r", 0)
                        weight = s.get("weight") or s.get("w", 0)
                        if isinstance(weight, dict): weight = weight.get("kg") or weight.get("val") or 0
                        sets.append({"reps": int(reps or 0), "weight": float(weight or 0)})
                    exercises.append({"name": name, "sets": sets})
        for n in walk(blob, lambda k: "duration" in k or "durationMinutes" in k):
            duration_minutes = duration_minutes or n.get("duration") or n.get("durationMinutes")
        for n in walk(blob, lambda k: "date" in k or "startTime" in k or "start_date" in k):
            wdate = wdate or n.get("date") or n.get("startTime") or n.get("start_date")

    if not wdate:
        meta_date = soup.find("meta", attrs={"property": "article:published_time"}) or soup.find("time")
        if meta_date: wdate = meta_date.get("content") or meta_date.get_text(strip=True)

    total_volume = 0
    for e in exercises:
        for s in e["sets"]:
            total_volume += int((s.get("weight") or 0) * (s.get("reps") or 0))

    parsed_date = None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed_date = datetime.strptime(str(wdate)[:len(fmt)], fmt).date(); break
        except: continue

    return {
        "date": parsed_date or datetime.utcnow().date(),
        "duration_minutes": int(duration_minutes) if duration_minutes is not None else None,
        "total_volume": int(total_volume),
        "exercises": exercises,
        "raw_hint": "Parsed heuristically from Hevy share page"
    }

# ----- Routes -----
@app.post("/users", response_model=UserOut)
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    return get_or_create_user(db, name=body.name,
                              protein_target=body.protein_target,
                              nutrition_provider=body.nutrition_provider,
                              nutrition_api_base=body.nutrition_api_base,
                              nutrition_api_key=body.nutrition_api_key)

@app.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.post("/ingest/hevy", response_model=HevyWorkoutOut)
def ingest_hevy(body: HevyIn, db: Session = Depends(get_db)):
    user = get_or_create_user(db, name=body.user_name)
    parsed = parse_hevy_share(body.url)
    w = create_workout(db, user_id=user.id, dt=parsed["date"],
                       duration=parsed.get("duration_minutes"),
                       volume=parsed.get("total_volume"),
                       url=body.url, raw=parsed)
    return HevyWorkoutOut(user_name=user.name, date=w.date,
                          duration_minutes=w.duration_minutes,
                          total_volume=w.total_volume)

@app.post("/ingest/nutrition/manual")
def ingest_nutrition_manual(body: NutritionManualIn, db: Session = Depends(get_db)):
    user = get_or_create_user(db, name=body.user_name)
    upsert_nutrition(db, user_id=user.id, dt=body.date,
                     protein_g=body.protein_g, calories=body.calories,
                     carbs_g=body.carbs_g, fat_g=body.fat_g, source="manual")
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
        upsert_nutrition(db, user_id=user.id, dt=dt, protein_g=protein,
                         calories=calories, carbs_g=carbs, fat_g=fat, source="csv")
        n += 1
    return {"ok": True, "rows": n}
