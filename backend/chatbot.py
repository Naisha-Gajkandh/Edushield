"""Academic improvement chatbot - restricted to study/academic topics only."""
import re

ACADEMIC_KEYWORDS = [
    "study", "studying", "grade", "grades", "cgpa", "gpa", "exam", "exams", "attend", "attendance",
    "improve", "improving", "learn", "learning", "subject", "subjects", "course", "courses",
    "help", "struggling", "fail", "failing", "pass", "passing", "motivation", "motivated",
    "time management", "schedule", "focus", "concentration", "stress", "anxious", "anxiety",
    "assignment", "project", "homework", "lecture", "class", "semester",
    "performance", "score", "marks", "result", "results", "advice", "tip", "tips",
    "dropout", "drop out", "risk", "at risk", "probation", "suspension",
    "counselor", "counseling", "academic", "university", "college", "program"
]

OFF_TOPIC_RESPONSE = (
    "I can only help with questions about academic improvement. "
    "Please ask about studying, grades, attendance, time management, or how to improve your performance. "
    "For other topics, please contact your counselor."
)

RESPONSES = {
    "study": [
        "Try the Pomodoro technique: 25 min focus, 5 min break. Build up gradually.",
        "Create a study schedule. Study at the same time daily to build a habit.",
        "Study in a quiet place. Reduce distractions like phone and social media.",
    ],
    "grade": [
        "Focus on understanding concepts, not memorizing. Practice past exams.",
        "Meet with your instructor during office hours for clarification on tough topics.",
        "Form study groups with classmates—teaching others reinforces your learning.",
    ],
    "attendance": [
        "Attendance directly impacts grades. Aim for 90%+—each class builds on the previous.",
        "Set reminders the night before. Prepare your bag and materials in advance.",
        "If you miss a class, get notes from a classmate and review before the next lecture.",
    ],
    "motivation": [
        "Set small, achievable goals. Celebrate each win to stay motivated.",
        "Remember your 'why'—why you chose this program. Write it down and read it when low.",
        "Connect with a study buddy or mentor for accountability.",
    ],
    "time": [
        "Use a planner. Block study time like appointments you can't miss.",
        "Prioritize: tackle hardest subjects when you're most alert.",
        "Limit social media. Set app limits—even 1 hour less daily = 7 hours/week for study.",
    ],
    "stress": [
        "Take short breaks. Walk, stretch, or breathe deeply every 45-60 minutes.",
        "Sleep 7-8 hours. Your brain consolidates learning during sleep.",
        "Talk to a counselor if stress feels overwhelming. It's okay to ask for help.",
    ],
    "cgpa": [
        "CGPA improves over time. Focus on doing better this semester than last.",
        "Identify your weakest subjects and allocate more study time to them.",
        "Check if your university offers grade replacement or extra credit opportunities.",
    ],
    "risk": [
        "You're taking a positive step by seeking help. Meet with your academic advisor soon.",
        "Create an action plan: attendance goals, study hours, and check-ins with a counselor.",
        "Many students recover from probation. Consistency and support make a big difference.",
    ],
    "default": [
        "Focus on consistent attendance and regular study sessions. Small steps lead to big improvements.",
        "Don't hesitate to reach out to your teachers and counselors. They want to help you succeed.",
        "Build a routine: same study time and place each day helps your brain get into focus mode.",
    ],
}


def is_academic_question(message: str) -> bool:
    """Check if the message is related to academic improvement."""
    text = message.lower().strip()
    if len(text) < 3:
        return False
    for kw in ACADEMIC_KEYWORDS:
        if kw in text:
            return True
    return False


def get_chat_response(message: str) -> str:
    """Get a restricted chatbot response. Only answers academic improvement questions."""
    if not is_academic_question(message):
        return OFF_TOPIC_RESPONSE

    text = message.lower()
    import random

    for key, replies in RESPONSES.items():
        if key != "default" and key in text:
            return random.choice(replies)
    return random.choice(RESPONSES["default"])
