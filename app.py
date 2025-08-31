from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder
from datetime import date, datetime
import os, io, csv, json, re
import httpx
from bs4 import BeautifulSoup

from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey, JSON, UniqueConstraint, BigInteger
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from sqlalchemy.exc import IntegrityError

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
app = FastAPI(title="Coach Ingestor API (single-file)", version="0.1.2")

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
    safe_raw = None
    if raw is not None:
        # Convert date/datetime (and any other non-JSON types) to strings
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

# ----- HTTP fetch headers (more browser-like) -----
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

# ----- Hevy parser (robust) -----
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
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for k in ("kg", "value", "val", "amount", "weight"):
            if k in x:
                return _num_from_any(x[k])
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
    looks_lb = ("weightlbs" in s or "lbs" in txt or " lb" in txt or unit in ("lb", "lbs", "pounds"))
    looks_kg = ("kg" in txt) or ("weightkg" in s) or (unit in ("kg", "kilogram", "kilograms"))
    if val is None:
        return 0.0
    if looks_lb and not looks_kg:
        return round(val * 0.45359237, 3)
    return float(val)

def parse_hevy_share(url: str) -> dict:
    # Short timeouts + retries + Referer so we fail fast and get clearer errors
    with httpx.Client(
        timeout=httpx.Timeout(8.0, connect=5.0, read=8.0),
        follow_redirects=True,
        headers=HTTP_HEADERS | {"Referer": "https://hevy.com/"},
        transport=httpx.HTTPTransport(retries=2),
    ) as client:
        r = client.get(url)
        r.raise_for_status()
        html = r.text

    soup = BeautifulSoup(html, "html.parser")

    data_sources = []
    nd = extract_next_data(html)
    if nd:
        data_sources.append(nd)

    for blob in re.findall(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, flags=re.S | re.I):
        try:
            data_sources.append(json.loads(blob))
        except Exception:
            pass

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

            if duration_minutes is None:
                if "durationMinutes" in node:
                    duration_minutes = int(_num_from_any(node.get("durationMinutes")) or 0)
                elif "workoutDuration" in node:
                    duration_minutes = int(_num_from_any(node.get("workoutDuration")) or 0)
                elif "durationSec" in node or "durationSeconds" in node:
                    secs = _num_from_any(node.get("durationSec") or node.get("durationSeconds"))
                    if secs is not None:
                        duration_minutes = int(round(secs / 60))
                elif "duration" in node and isinstance(node.get("duration"), (int, float, str, dict)):
                    duration_minutes = int(_num_from_any(node.get("duration")) or 0)

            if not wdate:
                wdate = node.get("date") or node.get("startTime") or node.get("startDate") or node.get("workoutDate")

    if not wdate:
        meta_date = soup.find("meta", attrs={"property": "article:published_time"}) or soup.find("time")
        if meta_date:
            wdate = meta_date.get("content") or meta_date.get_text(strip=True)

    total_volume = 0
    for e in exercises:
        for s in e["sets"]:
            total_volume += int(round((s.get("weight_kg") or 0) * (s.get("reps") or 0)))

    parsed_date = None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed_date = datetime.strptime(str(wdate)[:len(fmt)], fmt).date()
            break
        except Exception:
            continue

    return {
        "date": parsed_date or datetime.utcnow().date(),
        "duration_minutes": int(duration_minutes) if duration_minutes is not None else None,
        "total_volume": int(total_volume),
        "exercises": exercises,
        "raw_hint": "Parsed from Hevy share (weights normalized to kg)",
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
    # 1) ensure user exists
    user = get_or_create_user(db, name=body.user_name)

    # 2) parse the share page
    try:
        parsed = parse_hevy_share(body.url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Hevy fetch/parse failed: {type(e).__name__}: {e}")

    # 3) JSON-safe 'raw'
    safe_raw = jsonable_encoder(parsed, custom_encoder={
        date: lambda v: v.isoformat(),
        datetime: lambda v: v.isoformat(),
    })

    # 4) UPSERT: if (user, url) exists -> update; else insert
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
            # extremely rare race: re-fetch row and update
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

# --- tiny debug endpoints ---
@app.get("/debug/hevy")
def debug_hevy(url: str):
    try:
        parsed = parse_hevy_share(url)
        return {
            "ok": True,
            "date": parsed.get("date"),
            "duration_minutes": parsed.get("duration_minutes"),
            "total_volume": parsed.get("total_volume"),
            "exercise_count": len(parsed.get("exercises") or []),
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

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

@app.get("/debug/fetch")
def debug_fetch(url: str):
    try:
        with httpx.Client(
            timeout=httpx.Timeout(8.0, connect=5.0, read=8.0),
            follow_redirects=True,
            headers=HTTP_HEADERS | {"Referer": "https://hevy.com/"},
            transport=httpx.HTTPTransport(retries=2),
        ) as client:
            r = client.get(url)
            return {"ok": True, "status": r.status_code, "final_url": str(r.url), "length": len(r.text)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# --- Simple homepage (unchanged except version bump) ---
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
      <p class="muted">Paste your Hevy share link or log today’s protein. It’s quick.</p>

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
          <h3>Log protein for today</h3>
          <form id="protForm">
            <label for="protUser">Who are you?</label>
            <select id="protUser" required></select>

            <label for="protDate">Date</label>
            <input id="protDate" type="date" required />

            <label for="protG">Protein (g)</label>
            <input id="protG" type="number" min="0" step="1" placeholder="e.g. 180" required />

            <button type="submit">Save protein</button>
            <small class="muted">You can upload CSVs in the API docs later.</small>
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
        for (const id of ['hevyUser','protUser']) {
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

    document.getElementById('grafanaHref').href = 'http://<YOUR-SERVER>:3030/d/<UID>/<slug>?orgId=1&var-user=All&kiosk';
    document.getElementById('dashLink').href = document.getElementById('grafanaHref').href;
    document.getElementById('dashLink').setAttribute('target','_blank');

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
    loadUsers();
    </script>
  </body>
</html>
    """
