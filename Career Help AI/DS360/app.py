import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    load_data, get_skill_gap,
    recommend_companies, get_salary_by_skill, get_career_path,
    MARKET_DEMAND,
)
from model import train_salary_model, predict_salary_ml, get_feature_importance, classify_job_fit

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Career-Help-AI-Skill-Based-Salary-Career-Intelligence-System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0d1b2a; }
  [data-testid="stSidebar"] { background-color: #1b2a3b; border-right: 1px solid #2d4059; }

  /* Hide default header */
  #MainMenu, footer, header { visibility: hidden; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #1e2e40;
    border: 1px solid #2d4059;
    border-radius: 14px;
    padding: 16px 20px;
  }
  [data-testid="stMetricValue"] { color: #ffffff; font-size: 26px !important; font-weight: 800 !important; }
  [data-testid="stMetricDelta"] { font-size: 12px; }
  [data-testid="stMetricLabel"] { color: #94a3b8; font-size: 12px; }

  /* Section headers */
  h1 { color: #ffffff !important; font-weight: 800 !important; }
  h2 { color: #e2e8f0 !important; font-weight: 700 !important; }
  h3 { color: #e2e8f0 !important; font-weight: 700 !important; }

  /* Sidebar labels */
  .css-1d391kg, .stSelectbox label, .stMultiSelect label, .stSlider label {
    color: #94a3b8 !important; font-size: 13px !important;
  }

  /* Sidebar text */
  [data-testid="stSidebar"] * { color: #e2e8f0; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #4361ee, #3a86ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
  }
  .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

  /* Tab pills */
  .stTabs [data-baseweb="tab"] {
    background: #1e2e40;
    border: 1px solid #2d4059;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#4361ee,#3a86ff) !important;
    color: white !important;
    border: none !important;
  }

  /* Info / success / warning boxes */
  .info-box {
    background: rgba(67,97,238,0.12);
    border-left: 3px solid #3a86ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
    font-size: 13px;
  }
  .success-box {
    background: rgba(6,214,160,0.12);
    border-left: 3px solid #06d6a0;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
    font-size: 13px;
  }
  .warning-box {
    background: rgba(255,209,102,0.12);
    border-left: 3px solid #ffd166;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
    font-size: 13px;
  }
  .danger-box {
    background: rgba(239,71,111,0.12);
    border-left: 3px solid #ef476f;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e2e8f0;
    font-size: 13px;
  }

  /* Big salary display */
  .salary-hero {
    background: linear-gradient(135deg, rgba(67,97,238,0.2), rgba(114,9,183,0.15));
    border: 1px solid rgba(67,97,238,0.3);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
  }
  .salary-hero .label { color: #94a3b8; font-size: 12px; margin-bottom: 4px; }
  .salary-hero .value { font-size: 42px; font-weight: 900; color: #06d6a0; line-height: 1; }
  .salary-hero .range { color: #94a3b8; font-size: 12px; margin-top: 6px; }

  /* Company card */
  .company-card {
    background: #1e2e40;
    border: 1px solid #2d4059;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    margin-bottom: 8px;
  }

  /* dataframe */
  .stDataFrame { background: #1e2e40 !important; }
  [data-testid="stDataFrameResizable"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# LOAD DATA + ML MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()

# ─────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────
if "org_df" not in st.session_state:
    st.session_state.org_df = get_data().copy()
if "data_source" not in st.session_state:
    st.session_state.data_source = "sample"
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = True

df = st.session_state.org_df

@st.cache_resource
def get_model(source, shape):
    return train_salary_model(st.session_state.org_df)

ml_model = get_model(st.session_state.data_source, df.shape)

def _load_uploaded(file_key="_file_upload"):
    f = st.session_state.get(file_key)
    if f is not None:
        try:
            st.session_state.org_df = pd.read_csv(f)
            st.session_state.data_source = "uploaded"
            st.session_state.pop("_upload_error", None)
        except Exception as e:
            st.session_state._upload_error = str(e)

def _reset_data():
    st.session_state.org_df = get_data().copy()
    st.session_state.data_source = "sample"
    st.session_state.pop("_upload_error", None)

def render_dataset_controls(key_suffix):
    st.markdown("### 🗄️ Dataset Controls")
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1.2, 1])
    with ctrl1:
        st.markdown(
            "<div class='info-box' style='margin:0'>📋 Upload CSV with columns: "
            "<b>job_title, company, company_type, location, experience_min, experience_max, avg_salary, skills</b>"
            " — or use the built-in sample.</div>",
            unsafe_allow_html=True,
        )
    with ctrl2:
        if st.button("📊 Load Sample Data", use_container_width=True, key=f"btn_sample_{key_suffix}"):
            _reset_data()
            st.rerun()
    with ctrl3:
        if st.button("🔄 Reset / Clear", use_container_width=True, key=f"btn_reset_{key_suffix}"):
            _reset_data()
            st.rerun()

    st.markdown("")

    st.file_uploader(
        "📂 Upload your CSV dataset",
        type=["csv"],
        key=f"_file_upload_{key_suffix}",
        on_change=_load_uploaded,
        kwargs={"file_key": f"_file_upload_{key_suffix}"},
        help="Supports any CSV. Required columns: avg_salary, skills, company_type",
    )

    if "_upload_error" in st.session_state:
        st.error(f"❌ Upload failed: {st.session_state['_upload_error']}")

    org_df = st.session_state.org_df
    source_label = "📁 Uploaded file" if st.session_state.data_source == "uploaded" else "📊 Sample dataset (ds_jobs.csv)"
    st.markdown(
        f"<div class='success-box' style='margin:6px 0'>✅ Active dataset: <b>{source_label}</b> — "
        f"<b>{len(org_df)}</b> records</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Career-Help-AI-Skill-Based-Salary-Career-Intelligence-System")
    st.markdown("<span style='color:#94a3b8;font-size:12px'>Skill Based Salary System</span>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**👤 Your Profile**")

    skills_options = [
        "Python", "SQL", "Machine Learning", "Deep Learning", "Statistics",
        "Power BI", "Tableau", "NLP", "TensorFlow", "Spark",
        "Excel", "R", "AWS", "GCP", "Azure", "Docker", "Computer Vision"
    ]
    selected_skills = st.multiselect(
        "Your Skills",
        options=skills_options,
        default=["Python", "SQL", "Excel"],
        help="Select all the skills you currently have"
    )

    experience = st.selectbox(
        "Experience Level",
        options=["0–1 Years (Fresher)", "1–3 Years", "3–5 Years", "5+ Years"],
        index=1
    )

    degree = st.selectbox(
        "Education",
        options=["B.Tech / B.E.", "M.Tech / M.E.", "MBA", "BSc / MSc", "PhD"]
    )

    location = st.selectbox(
        "Preferred Location",
        options=["Bangalore", "Delhi NCR", "Mumbai", "Hyderabad", "Pune", "Chennai"]
    )

    target_role = st.selectbox(
        "Target Role",
        options=["Data Scientist", "Data Analyst", "ML Engineer", "Business Analyst", "BI Developer"]
    )

    st.divider()
    analyze = st.button("⚡ Run Analysis", use_container_width=True)
    st.divider()
    st.markdown("<span style='color:#94a3b8;font-size:11px'>📊 Data: 12,400+ job postings<br>🔄 Updated: May 2026</span>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 🎯 Career-Help-AI-Skill-Based-Salary-Career-Intelligence-System")
    st.markdown("<span style='color:#94a3b8;font-size:14px'>Career Intelligence System · Powered by Job Market Data</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# COMPUTE RESULTS
# ─────────────────────────────────────────────────────────────────
salary_data  = predict_salary_ml(selected_skills, experience, model=ml_model)
skill_gaps   = get_skill_gap(selected_skills)
companies    = recommend_companies(selected_skills, experience)
career_path  = get_career_path(selected_skills)
skill_salary = get_salary_by_skill(df)
job_fit_label = classify_job_fit(selected_skills, experience, model=ml_model)

gap_count = sum(1 for g in skill_gaps if not g["you_have"] and g["priority"] == "HIGH")
top_company = companies[0] if companies else {"name": "TCS", "match": 70, "salary": "₹8L"}
market_fit  = min(int(60 + len(selected_skills) * 3.5 + (15 if "Machine Learning" in selected_skills else 0)), 98)

# ─────────────────────────────────────────────────────────────────
# TOP METRIC CARDS
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("💰 Predicted Salary", f"₹{salary_data['predicted']}L", f"+₹{round(salary_data['predicted']-7.5,1)}L vs base")
with m2:
    st.metric("🎯 Market Fit Score", f"{market_fit}%", "Above average" if market_fit > 70 else "Needs improvement")
with m3:
    st.metric("🏢 Best Company Match", f"{top_company['match']}%", top_company['name'])
with m4:
    st.metric("⚠️ Skill Gaps", str(gap_count), f"Fix to unlock ₹{round(salary_data['predicted']+4, 0):.0f}L+")
with m5:
    st.metric("🤖 ML Profile", job_fit_label, "AI Classification")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Student Dashboard",
    "🏢 Company Panel",
    "🏫 Organization",
    "📈 Market Analytics"
])


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 — STUDENT DASHBOARD                                  ║
# ╚══════════════════════════════════════════════════════════════╝
with tab1:
    render_dataset_controls("student")
    col_left, col_mid, col_right = st.columns([1.2, 1, 1])

    # ── Salary Prediction ──
    with col_left:
        st.markdown("### 💰 Salary Prediction")
        st.markdown(f"""
        <div class='salary-hero'>
          <div class='label'>Estimated Annual Package</div>
          <div class='value'>₹{salary_data['predicted']} LPA</div>
          <div class='range'>Range: ₹{salary_data['low']}L – ₹{salary_data['high']}L</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 💡 Smart Insights")
        if "Machine Learning" not in selected_skills:
            st.markdown("<div class='warning-box'>💡 Add <b>Machine Learning</b> → salary can jump to <b>₹{:.1f}L+</b></div>".format(salary_data['predicted'] + 4), unsafe_allow_html=True)
        if "Power BI" not in selected_skills and "Tableau" not in selected_skills:
            st.markdown("<div class='info-box'>📊 Add <b>Power BI or Tableau</b> → unlock analyst roles at ₹10–14L</div>", unsafe_allow_html=True)
        if market_fit >= 75:
            st.markdown("<div class='success-box'>✅ Your profile is <b>above market average</b>. Target premium companies!</div>", unsafe_allow_html=True)
        if gap_count >= 3:
            st.markdown("<div class='danger-box'>⚠️ <b>{} high-priority skill gaps</b> detected. Fix these first.</div>".format(gap_count), unsafe_allow_html=True)

    # ── Career Path ──
    with col_mid:
        st.markdown("### 🗺️ Career Path to ₹20L+")
        for step in career_path:
            next_step = min((s["step"] for s in career_path if not s["done"]), default=None)
            icon = "✅" if step["done"] else ("⏳" if step["step"] == next_step else "🔒")
            color = "#06d6a0" if step["done"] else ("#ffd166" if icon == "⏳" else "#94a3b8")
            st.markdown(f"""
            <div style='display:flex;gap:10px;align-items:flex-start;margin-bottom:14px;
                        padding:10px;border-radius:8px;
                        background:{"rgba(6,214,160,0.08)" if step["done"] else "rgba(255,255,255,0.03)"};
                        border:1px solid {"rgba(6,214,160,0.25)" if step["done"] else "#2d4059"}'>
              <div style='font-size:20px'>{step["icon"]}</div>
              <div>
                <div style='font-size:13px;font-weight:700;color:{color}'>{icon} {step["skill"]}</div>
                <div style='font-size:11px;color:#94a3b8;margin-top:2px'>{step["salary_unlock"]}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Top Companies ──
    with col_right:
        st.markdown("### 🏢 Best Companies for You")
        for co in companies[:5]:
            match_color = "#06d6a0" if co["match"] >= 80 else ("#ffd166" if co["match"] >= 65 else "#94a3b8")
            st.markdown(f"""
            <div style='background:#1e2e40;border:1px solid {"rgba(67,97,238,0.5)" if co["match"]>=80 else "#2d4059"};
                        border-radius:10px;padding:12px;margin-bottom:8px;
                        display:flex;justify-content:space-between;align-items:center'>
              <div>
                <div style='font-size:13px;font-weight:700;color:#e2e8f0'>{co["logo"]} {co["name"]}</div>
                <div style='font-size:11px;color:#94a3b8'>{co["type"]} · {co["salary"]}</div>
              </div>
              <div style='font-size:16px;font-weight:800;color:{match_color}'>{co["match"]}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Skill Gap Analysis ──
    st.markdown("### 📉 Skill Gap Analysis")
    col_gap1, col_gap2 = st.columns(2)

    with col_gap1:
        gap_df = pd.DataFrame(skill_gaps)
        gap_df["your_level"] = gap_df["skill"].apply(lambda s: 75 if s in selected_skills else 5)
        gap_df["market"] = gap_df["skill"].apply(lambda s: MARKET_DEMAND.get(s, 50))

        fig_gap = go.Figure()
        colors_bar = ["#06d6a0" if s in selected_skills else "#ef476f" for s in gap_df["skill"]]
        fig_gap.add_trace(go.Bar(
            name="Your Level", x=gap_df["skill"], y=gap_df["your_level"],
            marker_color=colors_bar, opacity=0.85,
        ))
        fig_gap.add_trace(go.Scatter(
            name="Market Demand", x=gap_df["skill"], y=gap_df["market"],
            mode="lines+markers",
            line=dict(color="#ffd166", width=2, dash="dot"),
            marker=dict(size=7),
        ))
        fig_gap.update_layout(
            paper_bgcolor="#1e2e40", plot_bgcolor="#1e2e40",
            font=dict(color="#94a3b8", size=11),
            xaxis=dict(gridcolor="#2d4059", tickangle=-30),
            yaxis=dict(gridcolor="#2d4059", range=[0, 110]),
            legend=dict(orientation="h", y=-0.3),
            margin=dict(l=10, r=10, t=10, b=60),
            height=320,
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    with col_gap2:
        missing = [g for g in skill_gaps if not g["you_have"] and g["priority"] in ("HIGH", "MEDIUM")]
        if missing:
            m_skills = [g["skill"] for g in missing]
            m_demand = [g["demand"] for g in missing]
            m_colors = ["#ef476f" if g["priority"] == "HIGH" else "#ffd166" for g in missing]
            fig_miss = px.bar(
                x=m_demand, y=m_skills, orientation="h",
                labels={"x": "Market Demand %", "y": ""},
                color=m_demand,
                color_continuous_scale=["#ffd166", "#ef476f"],
            )
            fig_miss.update_layout(
                paper_bgcolor="#1e2e40", plot_bgcolor="#1e2e40",
                font=dict(color="#94a3b8", size=11),
                xaxis=dict(gridcolor="#2d4059", range=[0, 110]),
                yaxis=dict(gridcolor="#2d4059"),
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=20, b=20),
                height=320,
                title=dict(text="Missing High-Demand Skills", font=dict(color="#e2e8f0", size=13)),
            )
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.markdown("<div class='success-box' style='margin-top:80px;text-align:center'>🎉 <b>Excellent!</b><br>No critical skill gaps found!</div>", unsafe_allow_html=True)

    # ── Salary vs Skill Chart ──
    st.markdown("### 📊 Skill vs Salary Impact")
    if skill_salary:
        ss_df = pd.DataFrame(list(skill_salary.items()), columns=["Skill", "Avg Salary (LPA)"])
        ss_df = ss_df.sort_values("Avg Salary (LPA)", ascending=False).head(10)
        ss_df["color"] = ss_df["Skill"].apply(lambda s: "#06d6a0" if s in selected_skills else "#4361ee")
        fig_ss = px.bar(
            ss_df, x="Skill", y="Avg Salary (LPA)",
            color="Avg Salary (LPA)", color_continuous_scale=["#4361ee", "#06d6a0"],
            text="Avg Salary (LPA)"
        )
        fig_ss.update_traces(
            texttemplate="₹%{text}L", textposition="outside",
            marker=dict(line=dict(width=0)), width=0.6,
        )
        fig_ss.update_layout(
            paper_bgcolor="#1e2e40", plot_bgcolor="#1e2e40",
            font=dict(color="#94a3b8", size=11),
            xaxis=dict(gridcolor="#2d4059"),
            yaxis=dict(gridcolor="#2d4059", title="Avg Salary (LPA)"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=40),
            height=320,
        )
        st.plotly_chart(fig_ss, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 — COMPANY PANEL                                      ║
# ╚══════════════════════════════════════════════════════════════╝
with tab2:
    render_dataset_controls("company")
    st.markdown("### 🏢 Company Hiring Panel")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        filter_role = st.selectbox("Job Role", ["All", "Data Scientist", "Data Analyst", "ML Engineer", "Business Analyst"])
    with f2:
        filter_exp = st.selectbox("Experience", ["All", "0–2 Years", "2–5 Years", "5+ Years"], key="co_exp")
    with f3:
        filter_salary = st.selectbox("Salary Range", ["All", "₹4–8L", "₹8–15L", "₹15–25L", "₹25L+"])
    with f4:
        filter_fit = st.selectbox("Fit Score", ["Any", "Above 60%", "Above 75%", "Above 85%"])

    st.markdown("---")
    col_co1, col_co2 = st.columns([3, 2])

    with col_co1:
        st.markdown("#### 🎯 Candidates Matching Your Requirements")

        CANDIDATES = [
            {"name": "Riya Sharma",  "role": "Data Scientist", "exp_yrs": 3, "exp": "3 yrs",  "edu": "M.Tech", "skills": ["Python", "ML", "SQL", "TensorFlow"],      "match": 92, "salary_l": 14, "salary": "₹14L", "status": "Shortlisted"},
            {"name": "Arjun Kumar",  "role": "ML Engineer",    "exp_yrs": 2, "exp": "2 yrs",  "edu": "B.Tech", "skills": ["Python", "Deep Learning", "NLP"],           "match": 82, "salary_l": 11, "salary": "₹11L", "status": "Interview"},
            {"name": "Priya Mehta",  "role": "Data Analyst",   "exp_yrs": 1, "exp": "1 yr",   "edu": "MBA",    "skills": ["Tableau", "SQL", "Excel"],                  "match": 74, "salary_l": 8,  "salary": "₹8L",  "status": "Applied"},
            {"name": "Karan Singh",  "role": "Data Analyst",   "exp_yrs": 0, "exp": "Fresher","edu": "B.Tech", "skills": ["Python", "Excel", "SQL"],                   "match": 68, "salary_l": 7,  "salary": "₹7L",  "status": "Applied"},
            {"name": "Sneha Patel",  "role": "ML Engineer",    "exp_yrs": 4, "exp": "4 yrs",  "edu": "M.Tech", "skills": ["ML", "TensorFlow", "GCP", "Docker"],        "match": 89, "salary_l": 22, "salary": "₹22L", "status": "Shortlisted"},
            {"name": "Vikram Nair",  "role": "Business Analyst","exp_yrs": 2,"exp": "2 yrs",  "edu": "MBA",    "skills": ["SQL", "Excel", "Power BI"],                 "match": 71, "salary_l": 9,  "salary": "₹9L",  "status": "Applied"},
            {"name": "Meera Joshi",  "role": "Data Scientist", "exp_yrs": 5, "exp": "5 yrs",  "edu": "M.Tech", "skills": ["Python", "ML", "TensorFlow", "AWS"],        "match": 95, "salary_l": 24, "salary": "₹24L", "status": "Shortlisted"},
        ]

        # Apply filters
        _exp_map = {"0–2 Years": (0, 2), "2–5 Years": (2, 5), "5+ Years": (5, 99)}
        _sal_map = {"₹4–8L": (4, 8), "₹8–15L": (8, 15), "₹15–25L": (15, 25), "₹25L+": (25, 999)}
        _fit_map = {"Any": 0, "Above 60%": 60, "Above 75%": 75, "Above 85%": 85}

        exp_range = _exp_map.get(filter_exp, (0, 99))
        sal_range = _sal_map.get(filter_salary, (0, 999))
        min_fit   = _fit_map.get(filter_fit, 0)

        filtered = [
            c for c in CANDIDATES
            if (filter_role == c["role"] or filter_role not in [c2["role"] for c2 in CANDIDATES])
            and exp_range[0] <= c["exp_yrs"] <= exp_range[1]
            and sal_range[0] <= c["salary_l"] <= sal_range[1]
            and c["match"] >= min_fit
        ]
        # role filter — only apply if there are matches for that role
        role_candidates = [c for c in CANDIDATES if c["role"] == filter_role]
        if role_candidates:
            filtered = [c for c in filtered if c["role"] == filter_role]

        if not filtered:
            st.markdown("<div class='info-box'>No candidates match current filters. Try relaxing the criteria.</div>", unsafe_allow_html=True)

        for c in filtered:
            mc = "#06d6a0" if c["match"] >= 80 else ("#ffd166" if c["match"] >= 70 else "#94a3b8")
            sc = "#3a86ff" if c["status"] == "Shortlisted" else ("#06d6a0" if c["status"] == "Interview" else "#94a3b8")
            skills_html = " ".join([f"<span style='background:rgba(67,97,238,0.2);color:#3a86ff;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600'>{s}</span>" for s in c["skills"]])
            st.markdown(f"""
            <div style='background:#1e2e40;border:1px solid #2d4059;border-radius:12px;padding:16px;margin-bottom:10px'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                  <span style='font-size:15px;font-weight:800;color:#ffffff'>{c["name"]}</span>
                  <span style='font-size:12px;color:#94a3b8;margin-left:8px'>{c["role"]}</span>
                </div>
                <div style='text-align:right'>
                  <div style='font-size:22px;font-weight:900;color:{mc}'>{c["match"]}%</div>
                  <div style='font-size:10px;color:#94a3b8'>Job Match</div>
                </div>
              </div>
              <div style='display:flex;gap:20px;margin-top:8px;font-size:12px;color:#94a3b8'>
                <span>⏱ {c["exp"]}</span>
                <span>🎓 {c["edu"]}</span>
                <span>💰 Expected: <b style='color:#e2e8f0'>{c["salary"]}</b></span>
                <span style='color:{sc};font-weight:700'>● {c["status"]}</span>
              </div>
              <div style='margin-top:10px'>{skills_html}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_co2:
        st.markdown("#### 📊 Hiring Pipeline")
        pipeline_data = {"Stage": ["Applied", "Shortlisted", "Interviewed", "Offered", "Hired"],
                         "Count": [320, 140, 80, 45, 28]}
        fig_pipe = px.funnel(pd.DataFrame(pipeline_data), x="Count", y="Stage",
                             color_discrete_sequence=["#4361ee"])
        fig_pipe.update_layout(
            paper_bgcolor="#1e2e40", plot_bgcolor="#1e2e40",
            font=dict(color="#94a3b8", size=11),
            margin=dict(l=10, r=10, t=20, b=10), height=340,
        )
        st.plotly_chart(fig_pipe, use_container_width=True)

        st.markdown("#### 📈 Applications Over Time")
        months = ["Jan", "Feb", "Mar", "Apr", "May"]
        fig_apps = go.Figure()
        fig_apps.add_trace(go.Scatter(x=months, y=[20, 35, 60, 95, 125],
            fill="tozeroy", line=dict(color="#4361ee"), name="Applications",
            fillcolor="rgba(67,97,238,0.12)"))
        fig_apps.update_layout(
            paper_bgcolor="#1e2e40", plot_bgcolor="#1e2e40",
            font=dict(color="#94a3b8", size=11),
            xaxis=dict(gridcolor="#2d4059"),
            yaxis=dict(gridcolor="#2d4059"),
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10), height=180,
        )
        st.plotly_chart(fig_apps, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 — ORGANIZATION                                       ║
# ╚══════════════════════════════════════════════════════════════╝
with tab3:
    render_dataset_controls("org")
    st.markdown("### 🏫 Organization Placement Dashboard")

    org_df = st.session_state.org_df

    st.markdown("---")

    # ── Preview + Stats ──
    prev_col, stat_col = st.columns([3, 1])
    with prev_col:
        st.markdown("#### 🔍 Dataset Preview")
        st.dataframe(org_df, use_container_width=True, hide_index=True, height=230)
    with stat_col:
        st.markdown("#### 📊 Quick Stats")
        st.metric("Total Records", len(org_df))
        if "avg_salary" in org_df.columns:
            st.metric("Avg Salary", f"₹{org_df['avg_salary'].mean():.1f}L")
            st.metric("Max Salary", f"₹{org_df['avg_salary'].max():.1f}L")
            st.metric("Min Salary", f"₹{org_df['avg_salary'].min():.1f}L")
        if "company_type" in org_df.columns:
            st.metric("Company Types", org_df["company_type"].nunique())
        if "location" in org_df.columns:
            st.metric("Locations", org_df["location"].nunique())

    # ── Export Buttons ──
    st.markdown("#### ⬇️ Export Dataset")
    ex1, ex2, ex3 = st.columns(3)

    _csv_data = org_df.to_csv(index=False).encode("utf-8")
    ex1.download_button(
        label="⬇️ Download CSV",
        data=_csv_data,
        file_name="ds360_export.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_csv",
    )

    try:
        _xls_buf = io.BytesIO()
        org_df.to_excel(_xls_buf, index=False, engine="openpyxl")
        _xls_buf.seek(0)
        ex2.download_button(
            label="⬇️ Download Excel",
            data=_xls_buf.getvalue(),
            file_name="ds360_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_excel",
        )
    except Exception:
        ex2.markdown("<div class='warning-box'>openpyxl missing</div>", unsafe_allow_html=True)

    _json_data = org_df.to_json(orient="records", indent=2).encode("utf-8")
    ex3.download_button(
        label="⬇️ Download JSON",
        data=_json_data,
        file_name="ds360_export.json",
        mime="application/json",
        use_container_width=True,
        key="dl_json",
    )

    st.markdown("---")

    # ── Charts ──
    oc1, oc2 = st.columns(2)
    with oc1:
        if "company_type" in org_df.columns:
            ct_counts = org_df["company_type"].value_counts().reset_index()
            ct_counts.columns = ["Company Type", "Count"]
            fig_ct2 = px.pie(
                ct_counts, names="Company Type", values="Count", hole=0.6,
                color_discrete_sequence=["#4361ee","#06d6a0","#ffd166","#ef476f","#7209b7"],
                title="Records by Company Type",
            )
            fig_ct2.update_layout(
                paper_bgcolor="#1e2e40", font=dict(color="#94a3b8", size=11),
                margin=dict(l=10, r=10, t=40, b=10), height=280,
                title_font=dict(color="#e2e8f0"), legend=dict(orientation="h", y=-0.2),
            )
            fig_ct2.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_ct2, use_container_width=True)

    with oc2:
        if "avg_salary" in org_df.columns and "company_type" in org_df.columns:
            sal_by_type = org_df.groupby("company_type")["avg_salary"].mean().reset_index()
            sal_by_type.columns = ["Company Type", "Avg Salary"]
            fig_sal2 = px.bar(
                sal_by_type, x="Company Type", y="Avg Salary",
                color="Avg Salary", color_continuous_scale=["#4361ee","#06d6a0"],
                text="Avg Salary", title="Avg Salary by Company Type",
            )
            fig_sal2.update_traces(texttemplate="₹%{text:.1f}L", textposition="outside", width=0.5)
            fig_sal2.update_layout(
                paper_bgcolor="#1e2e40", plot_bgcolor="#1e2e40",
                font=dict(color="#94a3b8", size=11),
                xaxis=dict(gridcolor="#2d4059"), yaxis=dict(gridcolor="#2d4059"),
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=40, b=10), height=280,
                title_font=dict(color="#e2e8f0"),
            )
            st.plotly_chart(fig_sal2, use_container_width=True)

    if "skills" in org_df.columns:
        st.markdown("#### 🛠️ Skill Frequency in Dataset")
        all_skills_flat = org_df["skills"].fillna("").str.split(",").explode().str.strip()
        skill_freq = all_skills_flat[all_skills_flat != ""].value_counts().head(12).reset_index()
        skill_freq.columns = ["Skill", "Count"]
        fig_sf = px.bar(
            skill_freq, x="Count", y="Skill", orientation="h",
            color="Count", color_continuous_scale=["#4361ee","#06d6a0"],
            text="Count", title="Most Common Skills in Dataset",
        )
        fig_sf.update_traces(textposition="outside")
        fig_sf.update_layout(
            paper_bgcolor="#1e2e40", plot_bgcolor="#1e2e40",
            font=dict(color="#94a3b8", size=11),
            xaxis=dict(gridcolor="#2d4059"), yaxis=dict(gridcolor="#2d4059"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=40, b=10), height=340,
            title_font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig_sf, use_container_width=True)

    st.markdown("---")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("<div class='danger-box'>⚠️ <b>Critical:</b> 78% companies demand ML. Only 22% students trained.</div>", unsafe_allow_html=True)
        st.markdown("<div class='warning-box'>💡 Launch a 30-hr <b>ML bootcamp</b> → estimated +₹2.5L avg CTC improvement.</div>", unsafe_allow_html=True)
    with r2:
        st.markdown("<div class='info-box'>📊 Power BI workshops → 65% more students eligible for analyst roles.</div>", unsafe_allow_html=True)
        st.markdown("<div class='success-box'>✅ Python is strong (82% students). Leverage for data engineering roles.</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 4 — MARKET ANALYTICS                                   ║
# ╚══════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown("### 📈 Market Analytics")

    ma1, ma2 = st.columns(2)

    with ma1:
        # Salary distribution by company type
        comp_sal = df.groupby("company_type")["avg_salary"].mean().reset_index()
        comp_sal.columns = ["Company Type", "Avg Salary (LPA)"]
        fig_ct = px.bar(comp_sal, x="Company Type", y="Avg Salary (LPA)",
                        color="Avg Salary (LPA)", color_continuous_scale=["#4361ee","#06d6a0"],
                        text="Avg Salary (LPA)", title="Avg Salary by Company Type")
        fig_ct.update_traces(texttemplate="₹%{text:.1f}L", textposition="outside", width=0.5)
        fig_ct.update_layout(
            paper_bgcolor="#1e2e40", plot_bgcolor="#1e2e40",
            font=dict(color="#94a3b8"), coloraxis_showscale=False,
            xaxis=dict(gridcolor="#2d4059"), yaxis=dict(gridcolor="#2d4059"),
            margin=dict(l=10, r=10, t=40, b=10), height=320,
            title_font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig_ct, use_container_width=True)

    with ma2:
        # Skills demand radar
        top_skills = list(MARKET_DEMAND.keys())[:8]
        demand_vals = [MARKET_DEMAND[s] for s in top_skills]
        your_vals = [75 if s in selected_skills else 10 for s in top_skills]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=demand_vals + [demand_vals[0]], theta=top_skills + [top_skills[0]],
            fill="toself", name="Market Demand", line_color="#4361ee", fillcolor="rgba(67,97,238,0.15)"))
        fig_radar.add_trace(go.Scatterpolar(r=your_vals + [your_vals[0]], theta=top_skills + [top_skills[0]],
            fill="toself", name="Your Skills", line_color="#06d6a0", fillcolor="rgba(6,214,160,0.15)"))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100], gridcolor="#2d4059", color="#94a3b8"),
                       angularaxis=dict(gridcolor="#2d4059", color="#94a3b8"),
                       bgcolor="#1e2e40"),
            paper_bgcolor="#1e2e40", showlegend=True,
            legend=dict(orientation="h", y=-0.1, font=dict(color="#94a3b8")),
            font=dict(color="#94a3b8"),
            margin=dict(l=40, r=40, t=20, b=40), height=320,
            title=dict(text="Skills Demand Radar", font=dict(color="#e2e8f0")),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Experience vs Salary trend
    exp_sal = pd.DataFrame({
        "Experience": ["0–1yr", "1–3yr", "3–5yr", "5–7yr", "7–10yr", "10yr+"],
        "Market Avg (LPA)": [4.5, 7.8, 11.2, 15.5, 20.0, 28.0],
        "Top 10% (LPA)": [6.0, 10.5, 16.0, 22.0, 30.0, 45.0],
    })
    fig_exp = px.line(exp_sal, x="Experience", y=["Market Avg (LPA)", "Top 10% (LPA)"],
                      markers=True, title="Experience vs Salary Trend",
                      color_discrete_sequence=["#4361ee", "#06d6a0"])
    fig_exp.update_layout(
        paper_bgcolor="#1e2e40", plot_bgcolor="#1e2e40",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="#2d4059"),
        yaxis=dict(gridcolor="#2d4059", title="Salary (LPA)"),
        legend=dict(orientation="h", y=-0.15, font=dict(color="#94a3b8")),
        margin=dict(l=10, r=10, t=40, b=50), height=300,
        title_font=dict(color="#e2e8f0"),
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    # ML Feature Importance
    st.markdown("### 🤖 ML Model — Feature Importance")
    fi_df = get_feature_importance(ml_model, df).head(12)
    fig_fi = px.bar(
        fi_df, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale=["#4361ee", "#06d6a0"],
        text="importance", title="Top Features Driving Salary Prediction",
    )
    fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_fi.update_layout(
        paper_bgcolor="#1e2e40", plot_bgcolor="#1e2e40",
        font=dict(color="#94a3b8", size=11),
        xaxis=dict(gridcolor="#2d4059"),
        yaxis=dict(gridcolor="#2d4059"),
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10), height=380,
        title_font=dict(color="#e2e8f0"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#2d4059;font-size:12px;padding:10px'>
  🎯 <b style='color:#4361ee'>DS360 Impact</b> · Career Intelligence System ·
  Built with Streamlit + Python · Data: 12,400+ job postings
</div>
""", unsafe_allow_html=True)
