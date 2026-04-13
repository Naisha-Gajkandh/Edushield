"""SQLAlchemy models for the student dropout prevention system."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.database import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, index=True)
    name = Column(String(200))
    email = Column(String(200))
    phone = Column(String(20))
    gender = Column(String(20))
    age = Column(Integer)
    program = Column(String(100))
    current_semester = Column(Integer)
    admission_year = Column(Integer)
    current_cgpa = Column(Float)
    previous_sgpa = Column(Float)
    credits_completed = Column(Integer)
    attendance = Column(Float)
    family_income = Column(Integer)
    raw_data = Column(JSON)  # Store original Excel row data
    risk_score = Column(Float, default=0.0)
    risk_level = Column(String(20), default="low")  # low, medium, high, critical
    counselor_id = Column(Integer, ForeignKey("counselors.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    counselor = relationship("Counselor", back_populates="students")
    appointments = relationship("Appointment", back_populates="student")
    interventions = relationship("Intervention", back_populates="student")
    messages = relationship("Message", foreign_keys="Message.student_id", back_populates="student")


class Counselor(Base):
    __tablename__ = "counselors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200))
    email = Column(String(200))
    specialization = Column(String(100))

    students = relationship("Student", back_populates="counselor")
    appointments = relationship("Appointment", back_populates="counselor")


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    counselor_id = Column(Integer, ForeignKey("counselors.id"))
    scheduled_at = Column(DateTime)
    status = Column(String(20), default="scheduled")  # scheduled, completed, cancelled, no_show
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="appointments")
    counselor = relationship("Counselor", back_populates="appointments")


class Intervention(Base):
    __tablename__ = "interventions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    plan_type = Column(String(100))
    description = Column(Text)
    target_date = Column(DateTime)
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="interventions")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    sender_type = Column(String(20))  # student, counselor, admin, system
    subject = Column(String(200))
    body = Column(Text)
    read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="messages")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer)
    alert_type = Column(String(50))
    message = Column(Text)
    risk_score = Column(Float)
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
