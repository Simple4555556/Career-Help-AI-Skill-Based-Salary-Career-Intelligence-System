import os
import pandas as pd
import numpy as np

_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ds_jobs.csv")

SKILL_SALARY_BOOST = {
    "Python": 0,
    "SQL": 0,
    "Excel": -1.0,
    "Statistics": 1.5,
    "Machine Learning": 4.0,
    "Deep Learning": 5.5,
    "NLP": 4.5,
    "TensorFlow": 4.0,
    "Power BI": 2.0,
    "Tableau": 2.0,
    "Spark": 3.5,
    "R": 1.0,
    "Computer Vision": 5.0,
    "Docker": 2.5,
    "AWS": 3.0,
    "GCP": 3.0,
    "Azure": 3.0,
}

EXP_MULTIPLIERS = {
    "0–1 Years (Fresher)": 1.0,
    "1–3 Years": 1.3,
    "3–5 Years": 1.65,
    "5+ Years": 2.1,
}

COMPANY_TYPES = {
    "Premium (Google, Amazon, Microsoft)": {"min_skills": 4, "premium_skills": ["Machine Learning", "Deep Learning", "NLP"], "salary_range": (18, 30)},
    "Startup (Swiggy, Zomato, CRED)": {"min_skills": 3, "premium_skills": ["Machine Learning", "Python"], "salary_range": (12, 22)},
    "Service (TCS, Infosys, Wipro)": {"min_skills": 2, "premium_skills": ["SQL", "Excel"], "salary_range": (6, 10)},
    "MNC (SAP, IBM, Adobe)": {"min_skills": 3, "premium_skills": ["Python", "Statistics"], "salary_range": (10, 22)},
}

MARKET_DEMAND = {
    "Machine Learning": 78,
    "Python": 95,
    "SQL": 88,
    "Power BI": 65,
    "Deep Learning": 58,
    "Tableau": 52,
    "Statistics": 60,
    "NLP": 45,
    "TensorFlow": 48,
    "Spark": 40,
    "AWS": 70,
    "Excel": 55,
    "R": 30,
    "Docker": 38,
}


def load_data():
    try:
        df = pd.read_csv(_DATA_PATH)
        return df
    except FileNotFoundError:
        return _generate_mock_data()


def _generate_mock_data():
    np.random.seed(42)
    skills_pool = ["Python", "SQL", "Machine Learning", "Deep Learning", "Power BI", "Tableau", "Statistics", "NLP"]
    rows = []
    for i in range(50):
        n_skills = np.random.randint(2, 6)
        skills = ",".join(np.random.choice(skills_pool, n_skills, replace=False))
        rows.append({
            "job_id": i + 1,
            "job_title": np.random.choice(["Data Scientist", "Data Analyst", "ML Engineer"]),
            "company": f"Company_{i}",
            "company_type": np.random.choice(["Premium", "Service", "Startup", "MNC"]),
            "location": np.random.choice(["Bangalore", "Delhi", "Mumbai", "Hyderabad"]),
            "experience_min": np.random.randint(0, 4),
            "experience_max": np.random.randint(3, 8),
            "avg_salary": round(np.random.uniform(6, 25), 1),
            "skills": skills,
        })
    return pd.DataFrame(rows)


def predict_salary(skills: list[str], experience_label: str) -> dict:
    base = 7.5
    skill_boost = sum(SKILL_SALARY_BOOST.get(s, 0.5) for s in skills)
    exp_mult = EXP_MULTIPLIERS.get(experience_label, 1.0)
    predicted = (base + skill_boost) * exp_mult
    predicted = round(predicted, 1)
    low = round(predicted * 0.85, 1)
    high = round(predicted * 1.25, 1)
    return {"predicted": predicted, "low": low, "high": high}


def get_skill_gap(user_skills: list[str]) -> list[dict]:
    gaps = []
    for skill, demand in sorted(MARKET_DEMAND.items(), key=lambda x: -x[1]):
        has = skill in user_skills
        gaps.append({
            "skill": skill,
            "demand": demand,
            "you_have": has,
            "priority": "HIGH" if (not has and demand >= 65) else ("MEDIUM" if (not has and demand >= 45) else "LOW"),
        })
    return gaps


def recommend_companies(skills: list[str], experience_label: str) -> list[dict]:
    recommendations = []
    exp_years = {"0–1 Years (Fresher)": 0, "1–3 Years": 2, "3–5 Years": 4, "5+ Years": 6}
    exp = exp_years.get(experience_label, 1)
    skill_set = set(skills)

    COMPANIES = [
        {"name": "Google", "type": "Premium", "salary": "₹20–30L", "logo": "🔵", "req_skills": ["Python", "Machine Learning"], "min_exp": 2},
        {"name": "Amazon", "type": "Premium", "salary": "₹18–25L", "logo": "🟠", "req_skills": ["Python", "SQL", "Machine Learning"], "min_exp": 1},
        {"name": "Microsoft", "type": "Premium", "salary": "₹18–26L", "logo": "🔷", "req_skills": ["Python", "Machine Learning"], "min_exp": 2},
        {"name": "Swiggy", "type": "Startup", "salary": "₹12–18L", "logo": "🟠", "req_skills": ["Python", "SQL"], "min_exp": 1},
        {"name": "Zomato", "type": "Startup", "salary": "₹10–16L", "logo": "🔴", "req_skills": ["Python", "SQL"], "min_exp": 0},
        {"name": "Flipkart", "type": "Startup", "salary": "₹14–20L", "logo": "🟡", "req_skills": ["Python", "Machine Learning"], "min_exp": 1},
        {"name": "TCS", "type": "Service", "salary": "₹6–10L", "logo": "🔵", "req_skills": ["SQL", "Excel"], "min_exp": 0},
        {"name": "Infosys", "type": "Service", "salary": "₹7–11L", "logo": "🟤", "req_skills": ["SQL", "Python"], "min_exp": 0},
        {"name": "IBM", "type": "MNC", "salary": "₹10–18L", "logo": "🔵", "req_skills": ["Python", "SQL"], "min_exp": 1},
        {"name": "SAP", "type": "MNC", "salary": "₹14–22L", "logo": "🔷", "req_skills": ["Python", "Statistics"], "min_exp": 2},
    ]

    for co in COMPANIES:
        matched_req = sum(1 for s in co["req_skills"] if s in skill_set)
        match_pct = int((matched_req / max(len(co["req_skills"]), 1)) * 100)
        exp_ok = exp >= co["min_exp"]
        score = match_pct * (1.1 if exp_ok else 0.8)
        recommendations.append({**co, "match": min(int(score), 98), "exp_ok": exp_ok})

    recommendations.sort(key=lambda x: -x["match"])
    return recommendations


def get_salary_by_skill(df: pd.DataFrame) -> dict:
    result = {}
    for skill, _ in MARKET_DEMAND.items():
        mask = df["skills"].fillna("").str.contains(skill)
        if mask.sum() > 0:
            result[skill] = round(df[mask]["avg_salary"].mean(), 1)
    return result


def get_career_path(skills: list[str]) -> list[dict]:
    path = [
        {"step": 1, "skill": "Python + SQL", "done": "Python" in skills and "SQL" in skills, "salary_unlock": "₹6–9L", "icon": "🐍"},
        {"step": 2, "skill": "Excel + Statistics", "done": "Excel" in skills or "Statistics" in skills, "salary_unlock": "+₹1.5L", "icon": "📊"},
        {"step": 3, "skill": "Machine Learning", "done": "Machine Learning" in skills, "salary_unlock": "+₹4L → ₹14L", "icon": "🤖"},
        {"step": 4, "skill": "Deep Learning / NLP", "done": "Deep Learning" in skills or "NLP" in skills, "salary_unlock": "+₹5L → ₹18L", "icon": "🧠"},
        {"step": 5, "skill": "Cloud + MLOps", "done": any(s in skills for s in ["AWS", "GCP", "Azure", "Docker"]), "salary_unlock": "+₹5L → ₹22L+", "icon": "☁️"},
    ]
    return path
