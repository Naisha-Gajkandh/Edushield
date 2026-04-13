"""FastAPI backend for Student Dropout Prevention System."""
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import Optional
import io
import json

from backend.database import get_db
from backend.models import Student, Counselor, Appointment, Intervention, Message, Alert
from backend.config import settings
from ml.risk_model import predict_risk

app = FastAPI(title="Student Dropout Prevention API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve student app at root (login → results → chatbot)
@app.get("/")
async def serve_student_app():
    """Serve the student app. Open http://localhost:8000/ for the UI."""
    path = Path(__file__).resolve().parent.parent / "student_app.html"
    return FileResponse(path)


@app.get("/admin")
async def serve_admin_dashboard():
    """Admin/faculty dashboard at /admin"""
    path = Path(__file__).resolve().parent.parent / "dashboard.html"
    return FileResponse(path)


# ============ Dashboard & Risk ============

@app.get("/api/dashboard/summary")
async def get_dashboard_summary(db: AsyncSession = Depends(get_db)):
    """Risk Score Dashboard - real-time dropout risk overview."""
    total = await db.scalar(select(func.count(Student.id)))
    critical = await db.scalar(select(func.count(Student.id)).where(Student.risk_level == "critical"))
    high = await db.scalar(select(func.count(Student.id)).where(Student.risk_level == "high"))
    medium = await db.scalar(select(func.count(Student.id)).where(Student.risk_level == "medium"))
    low = await db.scalar(select(func.count(Student.id)).where(Student.risk_level == "low"))
    unacked_alerts = await db.scalar(select(func.count(Alert.id)).where(Alert.acknowledged == False))
    return {
        "total_students": total,
        "critical_risk": critical,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "unacknowledged_alerts": unacked_alerts,
    }


@app.get("/api/students/at-risk")
async def get_at_risk_students(
    level: Optional[str] = Query(None, description="Filter by risk level"),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List students by risk level."""
    q = select(Student).order_by(Student.risk_score.desc())
    if level:
        q = q.where(Student.risk_level == level)
    q = q.limit(limit)
    result = await db.execute(q)
    students = result.scalars().all()
    return [
        {
            "id": s.id,
            "student_id": s.student_id,
            "name": s.name,
            "program": s.program,
            "current_semester": s.current_semester,
            "current_cgpa": s.current_cgpa,
            "attendance": s.attendance,
            "risk_score": s.risk_score,
            "risk_level": s.risk_level,
        }
        for s in students
    ]


@app.get("/api/alerts")
async def get_alerts(
    acknowledged: Optional[bool] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Early Warning Alerts."""
    q = select(Alert).order_by(Alert.created_at.desc()).limit(limit)
    if acknowledged is not None:
        q = q.where(Alert.acknowledged == acknowledged)
    result = await db.execute(q)
    alerts = result.scalars().all()
    return [
        {
            "id": a.id,
            "student_id": a.student_id,
            "alert_type": a.alert_type,
            "message": a.message,
            "risk_score": a.risk_score,
            "acknowledged": a.acknowledged,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in alerts
    ]


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    a = await db.get(Alert, alert_id)
    if not a:
        raise HTTPException(404, "Alert not found")
    a.acknowledged = True
    await db.commit()
    return {"ok": True}


def _get_model_path():
    from pathlib import Path
    return str(Path(__file__).resolve().parent.parent / "ml_models" / "risk_model.joblib")


# ============ Student Management ============

@app.get("/api/students")
async def list_students(
    program: Optional[str] = None,
    risk_level: Optional[str] = None,
    search: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Student profiles with filters."""
    q = select(Student)
    if program:
        q = q.where(Student.program == program)
    if risk_level:
        q = q.where(Student.risk_level == risk_level)
    if search:
        q = q.where(
            (Student.name.ilike(f"%{search}%"))
            | (Student.student_id.ilike(f"%{search}%"))
            | (Student.email.ilike(f"%{search}%"))
        )
    q = q.offset(skip).limit(limit).order_by(Student.risk_score.desc())
    result = await db.execute(q)
    students = result.scalars().all()
    return [
        {
            "id": s.id,
            "student_id": s.student_id,
            "name": s.name,
            "email": s.email,
            "gender": s.gender,
            "age": s.age,
            "program": s.program,
            "current_semester": s.current_semester,
            "current_cgpa": s.current_cgpa,
            "attendance": s.attendance,
            "risk_score": s.risk_score,
            "risk_level": s.risk_level,
        }
        for s in students
    ]


@app.get("/api/student/login")
async def student_login(
    student_id: str = Query(..., description="Student ID e.g. STU01001"),
    db: AsyncSession = Depends(get_db),
):
    """Student login - lookup by student_id. Returns student data if found."""
    result = await db.execute(select(Student).where(Student.student_id == student_id.strip().upper()))
    s = result.scalar_one_or_none()
    if not s:
        raise HTTPException(404, "Student not found. Please check your Student ID.")
    subject_scores = _compute_subject_scores(s)
    risk_score, risk_level = _adjusted_risk(s, subject_scores)
    return {
        "id": s.id,
        "student_id": s.student_id,
        "name": s.name,
        "email": s.email,
        "program": s.program,
        "current_semester": s.current_semester,
        "current_cgpa": s.current_cgpa,
        "attendance": s.attendance,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "raw_data": s.raw_data,
    }


# 6 subjects from dataset with credits (Creativity has fewer credits)
SUBJECTS = ["Python", "Java", "Data Science", "DBMS", "Digital Electronics", "Creativity and Creation"]
SUBJECT_CREDITS = {
    "Python": 4,
    "Java": 4,
    "Data Science": 4,
    "DBMS": 4,
    "Digital Electronics": 3,
    "Creativity and Creation": 2,
}


def _compute_subject_scores(student) -> dict:
    """Get subject marks (0-100) from raw_data. Uses actual dataset columns."""
    raw = student.raw_data or {}
    result = {}
    for subj in SUBJECTS:
        v = raw.get(subj)
        if v is not None:
            try:
                result[subj] = int(float(v))
            except (ValueError, TypeError):
                result[subj] = 50
        else:
            result[subj] = 50  # fallback if column missing
    return result


def _adjusted_risk(student, subject_scores: dict) -> tuple[float, str]:
    """Compute risk from subject performance. Weak subjects (low marks) drive risk up."""
    if not subject_scores:
        return round(student.risk_score or 0, 1), student.risk_level or "low"

    # Per-subject contribution: low marks = high risk. Weight by credits.
    weighted_risk = 0
    total_weight = 0
    for subj, marks in subject_scores.items():
        cr = SUBJECT_CREDITS.get(subj, 4)
        # risk from this subject: 100 - marks (low marks = high risk)
        subj_risk = max(0, 100 - marks)
        weighted_risk += subj_risk * cr
        total_weight += cr
    subject_risk = weighted_risk / total_weight if total_weight else 50

    # Use subject-based risk as primary; blend with ML slightly for stability
    ml_score = student.risk_score or 50
    final_score = 0.7 * subject_risk + 0.3 * ml_score
    final_score = max(5, min(95, final_score))

    if final_score >= 75:
        level = "critical"
    elif final_score >= 50:
        level = "high"
    elif final_score >= 25:
        level = "medium"
    else:
        level = "low"
    return round(final_score, 1), level


@app.get("/api/student/{student_db_id}/insights")
async def get_student_insights(student_db_id: int, db: AsyncSession = Depends(get_db)):
    """Get subjects to focus on (max 3-4), good at, subject scores, and peer learning hints."""
    s = await db.get(Student, student_db_id)
    if not s:
        raise HTTPException(404, "Student not found")
    raw = s.raw_data or {}

    subject_scores = _compute_subject_scores(s)
    cgpa = s.current_cgpa or 0
    att = s.attendance or 0
    study_hrs = 0
    try:
        study_hrs = int(raw.get("study_hours", raw.get("How many hour do you study daily?", 0)) or 0)
    except (ValueError, TypeError):
        pass
    english_raw = str(raw.get("english_proficiency", raw.get("Status of your English language proficiency", "")) or "").lower().strip()
    skills = str(raw.get("skills", raw.get("What are the skills do you have ?", "")) or "")
    interested = str(raw.get("interested_area", raw.get("What is you interested area?", "")) or "")
    cocurricular = str(raw.get("cocurricular", raw.get("Are you engaged with any co-curriculum activities?", "")) or "").lower()

    # Focus on weak subjects only: below 60 for core, below 50 for Creativity
    focus_on = []
    for subj, score in subject_scores.items():
        threshold = 55 if SUBJECT_CREDITS.get(subj, 4) >= 3 else 50
        if score < threshold:
            focus_on.append(subj)
    if not focus_on:
        low = min(subject_scores.items(), key=lambda x: x[1])
        if low[1] < 75:
            focus_on.append(low[0])
    if att < 70:
        focus_on.append("Attendance")
    if study_hrs < 2:
        focus_on.append("Study habits")
    focus_on = focus_on[:6]

    # Good at: subjects with score >= 70
    good_at = [subj for subj, score in subject_scores.items() if score >= 70]
    if not good_at:
        best = max(subject_scores.items(), key=lambda x: x[1])
        good_at = [best[0]]

    risk_score, risk_level = _adjusted_risk(s, subject_scores)
    return {
        "focus_on": focus_on,
        "good_at": good_at,
        "subject_scores": subject_scores,
        "subjects": SUBJECTS,
        "subject_credits": SUBJECT_CREDITS,
        "is_at_risk": risk_level in ("high", "critical"),
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


@app.get("/api/student/{student_db_id}/peers")
async def get_peer_learners(
    student_db_id: int,
    subject: str = Query(..., description="Subject to find peers good at"),
    limit: int = Query(10, le=30),
    db: AsyncSession = Depends(get_db),
):
    """Find students who are good at a subject (for peer learning). Excludes current student."""
    current = await db.get(Student, student_db_id)
    if not current:
        raise HTTPException(404, "Student not found")
    subj = subject.strip()
    if subj not in SUBJECTS:
        raise HTTPException(400, f"Subject must be one of: {SUBJECTS}")

    result = await db.execute(select(Student).where(Student.id != student_db_id))
    all_students = result.scalars().all()
    peers = []
    for st in all_students:
        scores = _compute_subject_scores(st)
        if scores.get(subj, 0) >= 70:
            peers.append({
                "id": st.id,
                "student_id": st.student_id,
                "name": st.name,
                "program": st.program,
                "subject_score": scores.get(subj, 0),
            })
    peers.sort(key=lambda x: x["subject_score"], reverse=True)
    return peers[:limit]


@app.post("/api/chatbot")
async def chatbot_message(message: str = Body(..., embed=True)):
    """Academic improvement chatbot. Only responds to study/academic related questions."""
    from backend.chatbot import get_chat_response
    reply = get_chat_response(message)
    return {"reply": reply}


@app.get("/api/students/{student_id}")
async def get_student(student_id: int, db: AsyncSession = Depends(get_db)):
    """Comprehensive student profile."""
    s = await db.get(Student, student_id)
    if not s:
        raise HTTPException(404, "Student not found")
    return {
        "id": s.id,
        "student_id": s.student_id,
        "name": s.name,
        "email": s.email,
        "phone": s.phone,
        "gender": s.gender,
        "age": s.age,
        "program": s.program,
        "current_semester": s.current_semester,
        "admission_year": s.admission_year,
        "current_cgpa": s.current_cgpa,
        "previous_sgpa": s.previous_sgpa,
        "credits_completed": s.credits_completed,
        "attendance": s.attendance,
        "family_income": s.family_income,
        "risk_score": s.risk_score,
        "risk_level": s.risk_level,
        "raw_data": s.raw_data,
        "counselor_id": s.counselor_id,
    }


@app.post("/api/students/{student_id}/recalculate-risk")
async def recalculate_risk(student_id: int, db: AsyncSession = Depends(get_db)):
    """Recalculate ML risk score for a student."""
    s = await db.get(Student, student_id)
    if not s:
        raise HTTPException(404, "Student not found")
    model_path = _get_model_path()
    data = {"raw_data": s.raw_data or {}}
    score, level = predict_risk(data, model_path)
    s.risk_score = score
    s.risk_level = level
    await db.commit()
    return {"risk_score": score, "risk_level": level}


# ============ Analytics & Trends ============

@app.get("/api/analytics/trends")
async def get_trends(db: AsyncSession = Depends(get_db)):
    """Historical risk trends by program and semester."""
    q = select(
        Student.program,
        Student.current_semester,
        Student.risk_level,
        func.count(Student.id).label("count"),
    ).group_by(Student.program, Student.current_semester, Student.risk_level)
    result = await db.execute(q)
    rows = result.all()
    by_program = {}
    by_semester = {}
    for r in rows:
        key = f"{r.program}|{r.current_semester}"
        if key not in by_program:
            by_program[key] = {"program": r.program, "semester": r.current_semester, "low": 0, "medium": 0, "high": 0, "critical": 0}
        by_program[key][r.risk_level] = r.count
        if r.current_semester not in by_semester:
            by_semester[r.current_semester] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        by_semester[r.current_semester][r.risk_level] = by_semester[r.current_semester].get(r.risk_level, 0) + r.count
    return {"by_program_semester": list(by_program.values()), "by_semester": by_semester}


@app.get("/api/analytics/department")
async def get_department_stats(db: AsyncSession = Depends(get_db)):
    """Department-wise dropout risk comparison."""
    q = select(
        Student.program,
        func.count(Student.id).label("total"),
        func.avg(Student.risk_score).label("avg_risk"),
        func.sum(case((Student.risk_level == "critical", 1), else_=0)).label("critical"),
        func.sum(case((Student.risk_level == "high", 1), else_=0)).label("high"),
    ).group_by(Student.program)
    result = await db.execute(q)
    rows = result.all()
    return [
        {
            "program": r.program,
            "total": r.total,
            "avg_risk_score": round(float(r.avg_risk or 0), 1),
            "critical_count": r.critical or 0,
            "high_count": r.high or 0,
        }
        for r in rows
    ]


# ============ Counseling ============

@app.get("/api/counselors")
async def list_counselors(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Counselor))
    return [{"id": c.id, "name": c.name, "email": c.email, "specialization": c.specialization} for c in result.scalars().all()]


@app.get("/api/appointments")
async def list_appointments(
    student_id: Optional[int] = None,
    counselor_id: Optional[int] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(Appointment)
    if student_id:
        q = q.where(Appointment.student_id == student_id)
    if counselor_id:
        q = q.where(Appointment.counselor_id == counselor_id)
    if status:
        q = q.where(Appointment.status == status)
    q = q.order_by(Appointment.scheduled_at.desc())
    result = await db.execute(q)
    appointments = result.scalars().all()
    return [
        {
            "id": a.id,
            "student_id": a.student_id,
            "counselor_id": a.counselor_id,
            "scheduled_at": a.scheduled_at.isoformat() if a.scheduled_at else None,
            "status": a.status,
            "notes": a.notes,
        }
        for a in appointments
    ]


@app.post("/api/appointments")
async def create_appointment(
    student_id: int = Body(...),
    counselor_id: int = Body(...),
    scheduled_at: str = Body(...),
    notes: Optional[str] = Body(None),
    db: AsyncSession = Depends(get_db),
):
    ap = Appointment(
        student_id=student_id,
        counselor_id=counselor_id,
        scheduled_at=datetime.fromisoformat(scheduled_at.replace("Z", "+00:00")),
        notes=notes,
        status="scheduled",
    )
    db.add(ap)
    await db.flush()
    await db.commit()
    return {"id": ap.id, "status": "scheduled"}


@app.get("/api/interventions")
async def list_interventions(
    student_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(Intervention)
    if student_id:
        q = q.where(Intervention.student_id == student_id)
    q = q.order_by(Intervention.created_at.desc())
    result = await db.execute(q)
    interventions = result.scalars().all()
    return [
        {
            "id": i.id,
            "student_id": i.student_id,
            "plan_type": i.plan_type,
            "description": i.description,
            "target_date": i.target_date.isoformat() if i.target_date else None,
            "status": i.status,
        }
        for i in interventions
    ]


@app.post("/api/interventions")
async def create_intervention(
    student_id: int = Body(...),
    plan_type: str = Body(...),
    description: str = Body(...),
    target_date: Optional[str] = Body(None),
    db: AsyncSession = Depends(get_db),
):
    iv = Intervention(
        student_id=student_id,
        plan_type=plan_type,
        description=description,
        target_date=datetime.fromisoformat(target_date) if target_date else None,
        status="active",
    )
    db.add(iv)
    await db.flush()
    await db.commit()
    return {"id": iv.id, "status": "active"}


# ============ Messaging ============

@app.get("/api/messages")
async def list_messages(
    student_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(Message)
    if student_id:
        q = q.where(Message.student_id == student_id)
    q = q.order_by(Message.created_at.desc()).limit(100)
    result = await db.execute(q)
    msgs = result.scalars().all()
    return [
        {
            "id": m.id,
            "student_id": m.student_id,
            "sender_type": m.sender_type,
            "subject": m.subject,
            "body": m.body,
            "read": m.read,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in msgs
    ]


@app.post("/api/messages")
async def send_message(
    student_id: int = Body(...),
    sender_type: str = Body(...),
    subject: str = Body(...),
    content: str = Body(..., alias="body"),
    db: AsyncSession = Depends(get_db),
):
    m = Message(student_id=student_id, sender_type=sender_type, subject=subject, body=content)
    db.add(m)
    await db.flush()
    await db.commit()
    return {"id": m.id}


# ============ Reports & Export ============

@app.get("/api/reports/at-risk")
async def export_at_risk(
    format: str = Query("json", regex="^(json|csv)$"),
    db: AsyncSession = Depends(get_db),
):
    """Export at-risk students report."""
    q = select(Student).where(Student.risk_level.in_(["high", "critical"])).order_by(Student.risk_score.desc())
    result = await db.execute(q)
    students = result.scalars().all()
    data = [
        {
            "student_id": s.student_id,
            "name": s.name,
            "program": s.program,
            "cgpa": s.current_cgpa,
            "attendance": s.attendance,
            "risk_score": s.risk_score,
            "risk_level": s.risk_level,
        }
        for s in students
    ]
    if format == "csv":
        import csv
        output = io.StringIO()
        if data:
            w = csv.DictWriter(output, fieldnames=data[0].keys())
            w.writeheader()
            w.writerows(data)
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(output.getvalue(), media_type="text/csv")
    return data


@app.get("/health")
async def health():
    return {"status": "ok"}
