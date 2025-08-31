from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>LiftCrew – Submit</title>
    <!-- Clean, modern styling with zero build step -->
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
      <p class="muted">Paste your Hevy share link or log today’s protein. It’s quick.</p>

      <div class="grid">
        <!-- Hevy submit -->
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

        <!-- Protein log -->
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
        <!-- Want to embed? Replace the href + uncomment the iframe below -->
        <!--
        <div style="margin-top:1rem;">
          <iframe src="https://YOUR-GRAFANA-HOST/d/ABC123/your-board?orgId=1&kiosk"
                  width="100%" height="720" frameborder="0"></iframe>
        </div>
        -->
      </section>

      <footer>
        <p>Pro tip: save this page to your phone’s home screen for 1-tap logging.</p>
      </footer>
    </main>

    <script>
    const API = window.location.origin; // same host/port
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

    // If you have a Grafana URL, set it here for the button
    document.getElementById('grafanaHref').href = 'https://YOUR-GRAFANA-HOST/d/ABC123/your-board';

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

    // Defaults: set date to today; load users
    document.getElementById('protDate').valueAsDate = new Date();
    loadUsers();
    </script>
  </body>
</html>
    """
