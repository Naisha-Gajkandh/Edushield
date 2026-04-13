# Student Dropout Prevention System

A comprehensive system for real-time dropout risk assessment, student management, counseling, and analytics using your existing student performance dataset.

## Features

- **Risk Score Dashboard** – Real-time dropout risk overview by level (critical, high, medium, low)
- **Predictive Analytics** – ML-based risk prediction using attendance, grades, engagement
- **Early Warning Alerts** – Automated alerts when risk thresholds are crossed
- **Trend Analysis** – Historical visualization by program and semester
- **Student Management** – Profiles, attendance, grades, risk scores
- **Counseling** – Counselors, appointments, intervention plans
- **Analytics & Reporting** – Department stats, export to JSON/CSV

## Setup

### 1. Install Python Dependencies

```bash
cd c:\study\hackathon
pip install -r requirements.txt
```

### 2. Initialize Database

Load your `Students_Performance_data_set.xlsx` into the database and train the ML model:

```bash
python scripts/init_db.py
```

This will:

- Create SQLite database and tables
- Train the Random Forest risk model on your data
- Import ~1194 students with risk scores

### 3. Run Backend

```bash
python run_backend.py
```

API runs at `http://localhost:8000`

### 4. Open the App

- **Student portal**: Open `http://localhost:8000/` — Login with Student ID (e.g. STU01001), view results, subjects to focus on, and AI Study Coach
- **Admin dashboard**: Open `http://localhost:8000/admin` — Faculty/admin analytics and management

## Project Structure

```
hackathon/
├── Students_Performance_data_set.xlsx   # my dataset
├── requirements.txt
├── run_backend.py
├── dashboard.html                       # Single-page dashboard UI
├── backend/
│   ├── main.py                         # FastAPI app & all endpoints
│   ├── config.py
│   ├── database.py
│   ├── models.py                       # SQLAlchemy models
│   └── data_loader.py                  # Excel → DB loader
├── ml/
│   └── risk_model.py                   # ML risk prediction
├── scripts/
│   └── init_db.py                      # DB init & data load
└── ml_models/                          # Created by init_db
    └── risk_model.joblib
```

## API Endpoints


| Endpoint                                   | Description                 |
| ------------------------------------------ | --------------------------- |
| `GET /api/dashboard/summary`               | Risk counts, total students |
| `GET /api/students/at-risk`                | At-risk students list       |
| `GET /api/students`                        | Student list with filters   |
| `GET /api/students/{id}`                   | Student profile             |
| `POST /api/students/{id}/recalculate-risk` | Recalculate ML risk         |
| `GET /api/alerts`                          | Early warning alerts        |
| `POST /api/alerts/{id}/acknowledge`        | Acknowledge alert           |
| `GET /api/analytics/trends`                | Risk by program/semester    |
| `GET /api/analytics/department`            | Department comparison       |
| `GET /api/counselors`                      | List counselors             |
| `GET /api/appointments`                    | List appointments           |
| `GET /api/interventions`                   | Intervention plans          |
| `GET /api/messages`                        | Messages                    |
| `GET /api/reports/at-risk?format=json      | csv`                        |


## Tech Stack

- **Backend**: FastAPI, SQLAlchemy (async), SQLite
- **ML**: scikit-learn (Random Forest), pandas
- **Frontend**: Vanilla HTML/CSS/JS + Chart.js

