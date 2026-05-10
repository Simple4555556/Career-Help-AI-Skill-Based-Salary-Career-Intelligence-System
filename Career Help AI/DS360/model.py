import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from utils import load_data, SKILL_SALARY_BOOST, EXP_MULTIPLIERS

SKILLS_LIST = list(SKILL_SALARY_BOOST.keys())
EXP_LABELS = list(EXP_MULTIPLIERS.keys())

_mlb = MultiLabelBinarizer(classes=SKILLS_LIST)
_mlb.fit([SKILLS_LIST])


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    skill_lists = df["skills"].fillna("").str.split(",").apply(lambda x: [s.strip() for s in x])
    skill_matrix = pd.DataFrame(
        _mlb.transform(skill_lists),
        columns=_mlb.classes_,
        index=df.index,
    )
    exp_mid = ((df["experience_min"] + df["experience_max"]) / 2).fillna(2)
    feats = pd.concat([skill_matrix, exp_mid.rename("exp_mid")], axis=1)
    return feats


def train_salary_model(df: pd.DataFrame | None = None):
    if df is None:
        df = load_data()
    X = _build_features(df)
    y = df["avg_salary"]
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X, y)
    return model


def predict_salary_ml(skills: list[str], experience_label: str, model=None) -> dict:
    if model is None:
        model = train_salary_model()

    exp_map = {"0–1 Years (Fresher)": 0.5, "1–3 Years": 2.0, "3–5 Years": 4.0, "5+ Years": 6.5}
    exp_mid = exp_map.get(experience_label, 2.0)

    skill_vec = _mlb.transform([skills])
    feat_row = np.append(skill_vec[0], exp_mid).reshape(1, -1)
    col_names = list(_mlb.classes_) + ["exp_mid"]
    feat_df = pd.DataFrame(feat_row, columns=col_names)

    predicted = float(model.predict(feat_df)[0])
    predicted = round(predicted, 1)
    return {
        "predicted": predicted,
        "low": round(predicted * 0.85, 1),
        "high": round(predicted * 1.25, 1),
        "source": "ml_model",
    }


def evaluate_model(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = load_data()
    X = _build_features(df)
    y = df["avg_salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "r2": round(r2_score(y_test, preds), 3),
        "samples": len(df),
    }


def get_feature_importance(model=None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_data()
    if model is None:
        model = train_salary_model(df)
    X = _build_features(df)
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return importance


def classify_job_fit(skills: list[str], experience_label: str, model=None) -> str:
    salary = predict_salary_ml(skills, experience_label, model=model)["predicted"]
    if salary >= 20:
        return "Premium Candidate"
    elif salary >= 14:
        return "Strong Candidate"
    elif salary >= 9:
        return "Good Candidate"
    else:
        return "Entry Level"
