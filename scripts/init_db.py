"""Initialize database and load student data from Excel."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select, delete
from backend.database import engine, async_session, Base
from backend.models import Student, Counselor, Alert, Appointment, Intervention, Message
from backend.data_loader import load_and_transform_data, create_student_dict
from ml.risk_model import predict_risk, train_model


async def init_db():
    """Create tables and load initial data."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Clear existing data (allows re-running init_db without UNIQUE constraint errors)
    async with async_session() as session:
        await session.execute(delete(Alert))
        await session.execute(delete(Appointment))
        await session.execute(delete(Intervention))
        await session.execute(delete(Message))
        await session.execute(delete(Student))
        await session.execute(delete(Counselor))
        await session.commit()
        print("Cleared existing data.")

    base_path = ROOT
    excel_path = base_path / "Students_Performance_data_set.xlsx"
    model_path = base_path / "ml_models" / "risk_model.joblib"

    # Train ML model
    print("Training risk prediction model...")
    train_model(str(excel_path), str(model_path))
    print("Model trained.")

    # Load students from Excel
    df = load_and_transform_data(str(excel_path))

    async with async_session() as session:
        # Create sample counselors
        counselors = [
            Counselor(name="Dr. Sarah Mitchell", email="sarah.mitchell@uni.edu", specialization="Academic"),
            Counselor(name="Dr. James Chen", email="james.chen@uni.edu", specialization="Mental Health"),
            Counselor(name="Ms. Emma Wilson", email="emma.wilson@uni.edu", specialization="Career"),
        ]
        for c in counselors:
            session.add(c)
        await session.commit()

        # Load students
        for i, row in df.iterrows():
            sd = create_student_dict(row, i)
            risk_score, risk_level = predict_risk(sd, str(model_path))
            student = Student(
                student_id=sd["student_id"],
                name=sd["name"],
                email=sd["email"],
                gender=sd["gender"],
                age=sd["age"],
                program=sd["program"],
                current_semester=sd["current_semester"],
                admission_year=sd["admission_year"],
                current_cgpa=sd["current_cgpa"],
                previous_sgpa=sd["previous_sgpa"],
                credits_completed=sd["credits_completed"],
                attendance=sd["attendance"],
                family_income=sd["family_income"],
                raw_data=sd["raw_data"],
                risk_score=risk_score,
                risk_level=risk_level,
                counselor_id=(i % 3) + 1 if counselors else None,
            )
            session.add(student)

        await session.commit()
        print(f"Loaded {len(df)} students.")

        # Create sample high-risk alerts
        result = await session.execute(select(Student).where(Student.risk_level.in_(["high", "critical"])).limit(5))
        at_risk = result.scalars().all()
        for s in at_risk:
            session.add(Alert(
                student_id=s.id,
                alert_type="risk_threshold",
                message=f"Student {s.name} (CGPA: {s.current_cgpa}, Att: {s.attendance}%) crossed risk threshold",
                risk_score=s.risk_score,
            ))
        await session.commit()
        print("Database initialization complete.")


if __name__ == "__main__":
    asyncio.run(init_db())
