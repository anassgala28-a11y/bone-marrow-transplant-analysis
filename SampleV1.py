"""
BMT Clinical Decision Support — Single-Page Streamlit App
Run: streamlit run interface.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="BMT Predictor", page_icon="🧬",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    for p in ["models/xgboost.pkl","xgboost.pkl","../models/xgboost.pkl",
              "models/random_forest.pkl","models/lightgbm.pkl"]:
        if os.path.exists(p):
            try: return joblib.load(p), p
            except: pass
    return None, None
model, model_path = load_model()

# ── CSS + JS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Hard reset */
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;margin:0;padding:0;}
#MainMenu,footer,header,[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
[data-testid="stSidebar"]{display:none!important;}
.main .block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stMain"]{padding:0!important;}
[data-testid="stVerticalBlock"]{gap:0!important;padding:0!important;}
[data-testid="stHorizontalBlock"]{gap:0!important;}
.element-container{margin:0!important;padding:0!important;}
div[data-testid="stForm"]{background:transparent!important;border:none!important;padding:0!important;box-shadow:none!important;}

/* Sticky navbar */
.topbar{
  position:sticky;top:0;z-index:9999;
  background:#fff;border-bottom:2px solid #dbeafe;
  display:flex;align-items:center;padding:0 2rem;height:56px;
  box-shadow:0 2px 12px rgba(59,130,246,.1);
}
.topbar-logo{font-weight:800;font-size:1rem;color:#1d4ed8;margin-right:2rem;letter-spacing:-.02em;}
.topbar-logo span{color:#60a5fa;}
.topbar-links{display:flex;gap:.2rem;flex:1;}
.topbar-link{
  padding:.3rem .85rem;border-radius:6px;
  font-size:.78rem;font-weight:600;color:#64748b;
  text-decoration:none;transition:all .15s;cursor:pointer;
}
.topbar-link:hover,.topbar-link.active{background:#eff6ff;color:#1d4ed8;}
.topbar-badge{
  padding:.25rem .8rem;border-radius:20px;font-size:.68rem;font-weight:700;
  background:#f0fdf4;color:#15803d;border:1px solid #86efac;
}
.topbar-badge.off{background:#fef2f2;color:#b91c1c;border-color:#fca5a5;}

/* Section anchors */
.section-anchor{
  scroll-margin-top:70px;
  padding:2rem 2.5rem 0;
}
.section-anchor:first-of-type{padding-top:1.5rem;}

/* Section heading */
.sec-head{
  display:flex;align-items:center;gap:.6rem;
  font-size:1.05rem;font-weight:800;color:#1e3a5f;
  margin-bottom:1.2rem;letter-spacing:-.02em;
}
.sec-head .pill{
  background:#eff6ff;color:#1d4ed8;
  border:1px solid #bfdbfe;border-radius:20px;
  padding:2px 10px;font-size:.65rem;font-weight:700;
}

/* Card */
.card{
  background:#fff;border-radius:14px;
  border:1.5px solid #dbeafe;
  box-shadow:0 2px 12px rgba(59,130,246,.06);
  padding:1.5rem 1.7rem;margin-bottom:1.1rem;
}
.card-label{
  font-size:.65rem;font-weight:800;text-transform:uppercase;
  letter-spacing:.1em;color:#2563eb;
  margin-bottom:.9rem;padding-bottom:.6rem;
  border-bottom:2px solid #eff6ff;
}

/* Hero */
.hero{
  background:linear-gradient(120deg,#1d4ed8 0%,#3b82f6 100%);
  border-radius:16px;padding:2.8rem 3rem;
  position:relative;overflow:hidden;color:#fff;margin-bottom:1.5rem;
}
.hero::after{content:'🧬';position:absolute;right:2.5rem;top:50%;
  transform:translateY(-50%);font-size:7rem;opacity:.08;pointer-events:none;}
.hero-tag{display:inline-block;background:rgba(255,255,255,.18);
  color:#bfdbfe;border:1px solid rgba(255,255,255,.3);
  padding:3px 12px;border-radius:20px;font-size:.65rem;font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:.8rem;}
.hero h1{font-size:1.9rem;font-weight:800;line-height:1.2;
  letter-spacing:-.03em;margin:0 0 .6rem;}
.hero h1 em{font-style:normal;color:#93c5fd;}
.hero p{color:rgba(255,255,255,.76);font-size:.88rem;line-height:1.7;max-width:480px;}

/* Stats */
.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin-bottom:1.4rem;}
.stat-c{background:#fff;border-radius:12px;padding:1.1rem;text-align:center;
  border:1.5px solid #dbeafe;box-shadow:0 2px 8px rgba(59,130,246,.05);}
.stat-n{font-size:1.8rem;font-weight:800;color:#1d4ed8;line-height:1;letter-spacing:-.03em;}
.stat-l{font-size:.7rem;color:#94a3b8;margin-top:.2rem;font-weight:500;}

/* Steps */
.step-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin-bottom:1.4rem;}
.step-c{background:#fff;border-radius:12px;padding:1.2rem;
  border:1.5px solid #dbeafe;box-shadow:0 2px 8px rgba(59,130,246,.05);}
.step-n{width:34px;height:34px;border-radius:9px;
  background:linear-gradient(135deg,#1d4ed8,#3b82f6);
  color:#fff;font-size:.82rem;font-weight:800;
  display:flex;align-items:center;justify-content:center;margin-bottom:.7rem;}
.step-t{font-size:.8rem;font-weight:700;color:#1e3a5f;margin-bottom:.25rem;}
.step-d{font-size:.72rem;color:#94a3b8;line-height:1.5;}

/* Notice */
.notice{background:#fffbeb;border:1.5px solid #fde68a;border-radius:10px;
  padding:.8rem 1.1rem;font-size:.79rem;color:#78350f;font-weight:500;margin-bottom:1.2rem;}
.notice-b{background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:10px;
  padding:.75rem 1rem;font-size:.78rem;color:#1e40af;font-weight:500;margin-bottom:.9rem;}

/* Feature pills */
.fps{display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:1.2rem;}
.fp{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;
  border-radius:20px;padding:3px 11px;font-size:.68rem;font-weight:600;}

/* Divider */
.divider{border:none;border-top:2px solid #e2e8f0;margin:0 2.5rem 0;}

/* Result cards */
.res-ok{background:linear-gradient(135deg,#ecfdf5,#d1fae5);
  border:2px solid #6ee7b7;border-radius:14px;padding:2rem;text-align:center;}
.res-fail{background:linear-gradient(135deg,#fef2f2,#fee2e2);
  border:2px solid #fca5a5;border-radius:14px;padding:2rem;text-align:center;}
.res-icon{font-size:2.8rem;margin-bottom:.4rem;}
.res-v{font-size:1.05rem;font-weight:800;margin-bottom:.25rem;}
.res-ok .res-v{color:#065f46;} .res-fail .res-v{color:#7f1d1d;}
.res-p{font-size:2.7rem;font-weight:800;line-height:1;letter-spacing:-.04em;}
.res-ok .res-p{color:#059669;} .res-fail .res-p{color:#dc2626;}
.res-s{font-size:.72rem;color:#6b7280;margin-top:.3rem;font-weight:500;}

/* KPI */
.kpi{background:#fff;border:1.5px solid #dbeafe;border-radius:11px;
  padding:.95rem;text-align:center;box-shadow:0 2px 6px rgba(59,130,246,.05);}
.kpi-v{font-size:1.4rem;font-weight:800;color:#1d4ed8;letter-spacing:-.03em;}
.kpi-l{font-size:.64rem;color:#94a3b8;margin-top:.18rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.06em;}
.kpi-b{background:#f0fdf4;color:#15803d;border:1px solid #86efac;
  border-radius:4px;padding:2px 6px;font-size:.6rem;font-weight:700;
  display:inline-block;margin-top:.28rem;}

/* Footer */
.footer{background:#fff;border-top:2px solid #dbeafe;padding:.9rem 2rem;
  text-align:center;font-size:.72rem;color:#94a3b8;font-weight:500;margin-top:2rem;}
.footer strong{color:#1d4ed8;}

/* HIDE NAV TRIGGER BUTTONS */
[data-testid="stHorizontalBlock"]>div{margin:0!important;padding:0!important;}
[data-testid="stHorizontalBlock"] .stButton>button{
  height:0!important;min-height:0!important;padding:0!important;margin:0!important;
  border:none!important;background:none!important;color:transparent!important;
  font-size:0!important;overflow:hidden!important;visibility:hidden!important;}

/* FORM SUBMIT */
div[data-testid="stForm"] .stButton>button{
  visibility:visible!important;height:auto!important;min-height:2.4rem!important;
  padding:.5rem 1.1rem!important;background:#1d4ed8!important;
  border:none!important;color:#fff!important;font-size:.85rem!important;
  font-weight:700!important;border-radius:9px!important;
  box-shadow:0 3px 10px rgba(29,78,216,.25)!important;width:100%!important;}
div[data-testid="stForm"] .stButton>button:hover{background:#1e40af!important;}

/* Regular buttons */
.pb .stButton>button{
  visibility:visible!important;height:auto!important;min-height:2.1rem!important;
  padding:.45rem 1.3rem!important;background:#1d4ed8!important;
  border:none!important;color:#fff!important;font-size:.81rem!important;
  font-weight:700!important;border-radius:9px!important;
  box-shadow:0 3px 8px rgba(29,78,216,.2)!important;}
.pb .stButton>button:hover{background:#1e40af!important;}

/* Inputs */
label,.stSelectbox label,.stNumberInput label,
[data-testid="stWidgetLabel"] p{
  font-size:.73rem!important;font-weight:700!important;
  color:#374151!important;margin-bottom:.1rem!important;}
div[data-baseweb="select"]>div{
  background:#fff!important;border:1.5px solid #cbd5e1!important;
  border-radius:8px!important;font-size:.8rem!important;
  color:#111827!important;font-weight:500!important;}
div[data-baseweb="select"]>div:focus-within{
  border-color:#2563eb!important;
  box-shadow:0 0 0 3px rgba(37,99,235,.1)!important;}
[data-baseweb="popover"] *,[data-baseweb="menu"] *{
  background:#fff!important;color:#111827!important;
  font-size:.8rem!important;font-weight:500!important;}
[data-baseweb="menu"] li:hover{background:#eff6ff!important;color:#1d4ed8!important;}
input[type="number"]{
  background:#fff!important;border:1.5px solid #cbd5e1!important;
  border-radius:8px!important;font-size:.8rem!important;
  color:#111827!important;font-weight:500!important;}
input[type="number"]:focus{
  border-color:#2563eb!important;
  box-shadow:0 0 0 3px rgba(37,99,235,.1)!important;outline:none!important;}

/* Expander */
details{margin-bottom:.75rem!important;}
details>summary{
  background:#fff!important;border:1.5px solid #cbd5e1!important;
  border-radius:10px!important;padding:.65rem 1rem!important;
  font-size:.82rem!important;font-weight:700!important;color:#1e3a5f!important;
  cursor:pointer!important;list-style:none!important;
  display:flex!important;align-items:center!important;}
details>summary::-webkit-details-marker{display:none;}
details>summary::after{content:'▾';margin-left:auto;font-size:.82rem;color:#94a3b8;}
details[open]>summary::after{transform:rotate(180deg);display:inline-block;}
details>summary:hover{background:#eff6ff!important;border-color:#2563eb!important;color:#1d4ed8!important;}
details[open]>summary{border-radius:10px 10px 0 0!important;border-color:#2563eb!important;}
details .streamlit-expanderContent,
[data-testid="stExpander"]>div:last-child{
  background:#f8faff!important;border:1.5px solid #cbd5e1!important;
  border-top:none!important;border-radius:0 0 10px 10px!important;
  padding:1.1rem!important;}
details .streamlit-expanderContent *,
[data-testid="stExpander"]>div:last-child *{color:#111827!important;}
details .streamlit-expanderContent label,
[data-testid="stExpander"]>div:last-child label{color:#374151!important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  gap:.3rem;background:transparent!important;
  border-bottom:2px solid #dbeafe!important;padding-bottom:0!important;}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;border:none!important;
  font-size:.79rem!important;font-weight:700!important;color:#94a3b8!important;
  padding:.4rem .95rem!important;border-bottom:2px solid transparent!important;
  margin-bottom:-2px!important;}
.stTabs [aria-selected="true"]{color:#1d4ed8!important;border-bottom:2px solid #1d4ed8!important;}

/* Table */
[data-testid="stDataFrame"]{border-radius:11px!important;overflow:hidden!important;border:1.5px solid #dbeafe!important;}
[data-testid="stDataFrame"] th{background:#eff6ff!important;color:#1d4ed8!important;
  font-weight:800!important;font-size:.71rem!important;text-transform:uppercase!important;
  letter-spacing:.05em!important;padding:.5rem .7rem!important;border-bottom:2px solid #dbeafe!important;}
[data-testid="stDataFrame"] td{color:#111827!important;font-weight:500!important;
  font-size:.79rem!important;padding:.45rem .7rem!important;}
[data-testid="stDataFrame"] tr:nth-child(even) td{background:#f8faff!important;}

/* Metrics */
[data-testid="stMetric"]{background:#fff!important;border:1.5px solid #dbeafe!important;
  border-radius:10px!important;padding:.8rem .95rem!important;}
[data-testid="stMetricLabel"] p{font-size:.64rem!important;font-weight:700!important;
  color:#94a3b8!important;text-transform:uppercase!important;letter-spacing:.06em!important;}
[data-testid="stMetricValue"]{font-size:.95rem!important;font-weight:800!important;color:#1e3a5f!important;}
</style>

<script>
// Highlight active navbar link on scroll
window.addEventListener('scroll', function() {
  const sections = ['hero','input','results','shap','metrics'];
  const links = document.querySelectorAll('.topbar-link');
  let current = 'hero';
  sections.forEach(id => {
    const el = document.getElementById('sec-' + id);
    if (el && window.scrollY >= el.offsetTop - 80) current = id;
  });
  links.forEach(l => {
    l.classList.toggle('active', l.dataset.section === current);
  });
});
function scrollTo(id) {
  const el = document.getElementById('sec-' + id);
  if (el) el.scrollIntoView({behavior:'smooth'});
}
</script>
""", unsafe_allow_html=True)

# ── Encodings ─────────────────────────────────────────────────────────────
YN    = {"Yes":1,"No":0}
ABO   = {"O":-1,"A":0,"B":1,"AB":2}
DIS   = {"ALL":0,"AML":1,"chronic":2,"lymphoma":3,"nonmalignant":4}
ANT   = {"Not applicable":-1,"0":0,"1":1,"2":2,"3":3}
AGI   = {"0–5 years":0,"5–10 years":1,">10 years":2}
CMV   = {"-/- (both negative)":0,"+/- (donor+, recipient-)":1,
         "-/+ (donor-, recipient+)":2,"+/+ (both positive)":3}

# ── Matplotlib style helper ───────────────────────────────────────────────
def clean_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8faff")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#dbeafe")
    ax.spines["bottom"].set_color("#dbeafe")
    ax.tick_params(colors="#374151", labelsize=8.5)
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════
# STICKY NAVBAR
# ══════════════════════════════════════════════════════════════════════════
badge = ('bmt-badge ok',  '✔ Model Ready') if model else ('bmt-badge off', '⚠ No Model')
st.markdown(f"""
<div class="topbar">
  <div class="topbar-logo">🧬 BMT<span>.</span>AI</div>
  <div class="topbar-links">
    <span class="topbar-link active" data-section="hero"    onclick="scrollTo('hero')">🏠 Overview</span>
    <span class="topbar-link"       data-section="input"   onclick="scrollTo('input')">📋 Patient Input</span>
    <span class="topbar-link"       data-section="results" onclick="scrollTo('results')">📈 Prediction</span>
    <span class="topbar-link"       data-section="shap"    onclick="scrollTo('shap')">🔍 SHAP</span>
    <span class="topbar-link"       data-section="metrics" onclick="scrollTo('metrics')">📊 Metrics</span>
  </div>
  <span class="{badge[0]}">{badge[1]}</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — HERO / OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div id="sec-hero" class="section-anchor">', unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <div class="hero-tag">⚕️ Explainable AI · Pediatric Medicine · SHAP</div>
  <h1>Predict Transplant <em>Success.</em> Clearly.</h1>
  <p>A machine learning tool helping physicians assess the survival probability
  of pediatric bone marrow transplants with full SHAP transparency per prediction.</p>
</div>
<div class="stat-row">
  <div class="stat-c"><div class="stat-n">187</div><div class="stat-l">Patients in dataset</div></div>
  <div class="stat-c"><div class="stat-n">37</div><div class="stat-l">Clinical features</div></div>
  <div class="stat-c"><div class="stat-n">~87%</div><div class="stat-l">Best ROC-AUC</div></div>
  <div class="stat-c"><div class="stat-n">3</div><div class="stat-l">Models evaluated</div></div>
</div>
<div class="step-row">
  <div class="step-c"><div class="step-n">1</div><div class="step-t">Enter Patient Data</div><div class="step-d">Fill in donor, recipient and clinical variables below.</div></div>
  <div class="step-c"><div class="step-n">2</div><div class="step-t">Run Prediction</div><div class="step-d">XGBoost computes the transplant success probability.</div></div>
  <div class="step-c"><div class="step-n">3</div><div class="step-t">Read SHAP</div><div class="step-d">Understand which features drove the result.</div></div>
  <div class="step-c"><div class="step-n">4</div><div class="step-t">Review Metrics</div><div class="step-d">Validate model confidence and performance.</div></div>
</div>
<div class="notice">⚠️ <strong>Clinical Notice:</strong> This tool is a decision-support aid only. All predictions must be reviewed by a qualified physician before any clinical decision.</div>
""", unsafe_allow_html=True)
st.markdown('</div><hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — PATIENT INPUT
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div id="sec-input" class="section-anchor">', unsafe_allow_html=True)
st.markdown('<div class="sec-head">📋 Patient Input <span class="pill">Step 1</span></div>',
            unsafe_allow_html=True)

with st.form("main_form"):
    # Recipient
    st.markdown('<div class="card"><div class="card-label">👤 Recipient Information</div>',
                unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        Rg  = st.selectbox("Gender",          ["Male","Female"])
        Ra  = st.number_input("Age (years)",  0.0, 20.0, 8.0, 0.1)
        Rb  = st.number_input("Body Mass (kg)", 0.0, 120.0, 25.0, 0.5)
    with c2:
        Rd  = st.selectbox("Disease",         ["ALL","AML","chronic","nonmalignant","lymphoma"])
        Rdg = st.selectbox("Disease Group",   ["Malignant","Nonmalignant"])
        Rrg = st.selectbox("Risk Group",      ["Low","High"])
    with c3:
        Rabo= st.selectbox("Blood Type",      ["O","A","B","AB"])
        Rrh = st.selectbox("Rh Factor",       ["Positive","Negative"])
        Rcmv= st.selectbox("CMV Status",      ["Negative","Positive"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Donor
    st.markdown('<div class="card"><div class="card-label">🧑‍⚕️ Donor Information</div>',
                unsafe_allow_html=True)
    d1,d2,d3 = st.columns(3)
    with d1:
        Da   = st.number_input("Donor Age",   0.0, 80.0, 35.0, 0.1)
        Dabo = st.selectbox("Donor Blood Type",["O","A","B","AB"])
        Dcmv = st.selectbox("Donor CMV",      ["Negative","Positive"])
    with d2:
        Dsc  = st.selectbox("Stem Cell Source",["Bone Marrow","Peripheral Blood"])
        Dgm  = st.selectbox("Gender Match?",  ["No","Yes"])
        Dab  = st.selectbox("ABO Compatibility",["Incompatible","Compatible"])
    with d3:
        Dcmvs= st.selectbox("CMV Combination",["-/- (both negative)",
               "+/- (donor+, recipient-)","-/+ (donor-, recipient+)","+/+ (both positive)"])
        Dtr  = st.selectbox("Post-Relapse TX?",["No","Yes"])
        Drel = st.selectbox("Relapse?",       ["No","Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    # HLA
    with st.expander("🔬  HLA Matching & Disease Details"):
        h1,h2,h3 = st.columns(3)
        with h1:
            Hm   = st.selectbox("HLA Match",   ["Mismatch","Full Match"])
            Hmi  = st.selectbox("HLA Mismatch?",["No","Yes"])
            Han  = st.selectbox("Antigen mismatches",["Not applicable","0","1","2","3"])
        with h2:
            Hal  = st.selectbox("Allele mismatches",["Not applicable","0","1","2","3"])
            Hgr  = st.selectbox("HLA Group I?", ["No","Yes"])
            Hii  = st.selectbox("GvHD Grade II–IV compat.?",["No","Yes"])
        with h3:
            Ra10 = st.selectbox("Age > 10?",   ["No","Yes"])
            Rag  = st.selectbox("Age Group",   ["0–5 years","5–10 years",">10 years"])

    # Clinical
    with st.expander("📊  Clinical & Lab Parameters"):
        l1,l2,l3 = st.columns(3)
        with l1:
            Ccd34= st.number_input("CD34+ (×10⁶/kg)", 0.0, 200.0, 7.0, 0.1)
            Ccd3r= st.number_input("CD3/CD34 ratio",   0.0, 500.0, 5.0, 0.1)
            Ccd3 = st.number_input("CD3+ (×10⁸/kg)",  0.0, 100.0, 3.0, 0.01)
        with l2:
            Canc = st.number_input("ANC Recovery (days)", 0.0, 100.0, 15.0, 0.5)
            Cplt = st.number_input("Platelet Recovery (days)", 0.0, 200.0, 25.0, 0.5)
            Cagv = st.selectbox("Acute GvHD III–IV?", ["No","Yes"])
        with l3:
            Ccgv = st.selectbox("Extensive cGvHD?", ["No","Yes"])
            Ctag = st.number_input("Time to aGvHD (days; 1e6=never)",
                                   0.0, 1_000_000.0, 1_000_000.0, 1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍  Compute Prediction", use_container_width=True)

st.markdown('</div><hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC (runs when form submitted)
# ══════════════════════════════════════════════════════════════════════════
proba, pred, patient_df = None, None, None

if submitted:
    raw = {
        "Recipientgender":      1 if Rg=="Male" else 0,
        "Stemcellsource":       1 if Dsc=="Peripheral Blood" else 0,
        "Donorage":             Da,
        "Donorage35":           1 if Da>35 else 0,
        "IIIV":                 YN[Hii],
        "Gendermatch":          YN[Dgm],
        "DonorABO":             ABO[Dabo],
        "RecipientABO":         ABO[Rabo],
        "RecipientRh":          1 if Rrh=="Positive" else 0,
        "ABOmatch":             1 if Dab=="Compatible" else 0,
        "CMVstatus":            CMV[Dcmvs],
        "DonorCMV":             1 if Dcmv=="Positive" else 0,
        "RecipientCMV":         1 if Rcmv=="Positive" else 0,
        "Disease":              DIS[Rd],
        "Riskgroup":            1 if Rrg=="High" else 0,
        "Txpostrelapse":        YN[Dtr],
        "Diseasegroup":         1 if Rdg=="Malignant" else 0,
        "HLAmatch":             1 if Hm=="Full Match" else 0,
        "HLAmismatch":          YN[Hmi],
        "Antigen":              ANT[Han],
        "Alel":                 ANT[Hal],
        "HLAgrI":               YN[Hgr],
        "Recipientage":         Ra,
        "Recipientage10":       YN[Ra10],
        "Recipientageint":      AGI[Rag],
        "Relapse":              YN[Drel],
        "aGvHDIIIIV":           YN[Cagv],
        "extcGvHD":             YN[Ccgv],
        "CD34kgx10d6":          Ccd34,
        "CD3dCD34":             Ccd3r,
        "CD3dkgx10d8":          Ccd3,
        "Rbodymass":            Rb,
        "ANCrecovery":          Canc,
        "PLTrecovery":          Cplt,
        "time_to_aGvHD_III_IV": Ctag,
    }
    patient_df = pd.DataFrame([raw])

    if model:
        try:
            # Align columns — drop survival_time if present in model features
            if hasattr(model, "feature_names_in_"):
                needed = [c for c in model.feature_names_in_
                          if c != "survival_time" and c in patient_df.columns]
                patient_df = patient_df[needed]
            proba = float(model.predict_proba(patient_df)[0][1])
            pred  = int(model.predict(patient_df)[0])
        except Exception as e:
            st.error(f"Prediction error: {e}")
            proba, pred = 0.5, 0
    else:
        proba, pred = 0.73, 1   # demo fallback

    # Cache
    st.session_state["proba"]      = proba
    st.session_state["pred"]       = pred
    st.session_state["pdf"]        = patient_df
    st.session_state["show_results"] = True
    st.session_state["patient_summary"] = {
        "Recipient Gender": Rg,       "Age":          f"{Ra} yrs",
        "Body Mass":        f"{Rb} kg","Disease":      Rd,
        "Risk Group":       Rrg,      "Recipient ABO": Rabo,
        "Recipient CMV":    Rcmv,     "Donor Age":    f"{Da} yrs",
        "Donor ABO":        Dabo,     "Stem Cell Source": Dsc,
        "ABO Compat.":      Dab,      "HLA Match":    Hm,
        "Relapse":          Drel,     "CD34+":        f"{Ccd34} ×10⁶/kg",
        "ANC Recovery":     f"{Canc} days",
    }

# Retrieve from session
if st.session_state.get("show_results"):
    proba   = st.session_state["proba"]
    pred    = st.session_state["pred"]
    patient_df = st.session_state["pdf"]

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREDICTION RESULT (visible only after submit)
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div id="sec-results" class="section-anchor">', unsafe_allow_html=True)
st.markdown('<div class="sec-head">📈 Prediction Result <span class="pill">Step 2</span></div>',
            unsafe_allow_html=True)

if not st.session_state.get("show_results"):
    st.markdown('<div class="notice-b">ℹ️ Fill in the Patient Input form above and click <strong>Compute Prediction</strong> to see results here.</div>',
                unsafe_allow_html=True)
else:
    clr = "#059669" if pred == 1 else "#dc2626"
    left, right = st.columns([1, 1.4])

    with left:
        if pred == 1:
            st.markdown(f'<div class="res-ok"><div class="res-icon">✅</div>'
                        f'<div class="res-v">Transplant Likely Successful</div>'
                        f'<div class="res-p">{proba:.1%}</div>'
                        f'<div class="res-s">Survival probability estimate</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="res-fail"><div class="res-icon">⚠️</div>'
                        f'<div class="res-v">High Risk of Failure</div>'
                        f'<div class="res-p">{1-proba:.1%}</div>'
                        f'<div class="res-s">Failure probability estimate</div></div>',
                        unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge bar
        fig, ax = clean_fig(5.5, 0.85)
        ax.set_facecolor("#ffffff")
        ax.barh(0, proba,       color=clr,     height=0.36, alpha=0.9, zorder=2)
        ax.barh(0, 1-proba, left=proba, color="#eff6ff", height=0.36, zorder=2)
        ax.axvline(0.5, color="#93c5fd", lw=1.2, ls="--", zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(-0.4, 0.4)
        ax.set_xticks([0,.25,.5,.75,1])
        ax.set_xticklabels(["0%","25%","50%","75%","100%"], fontsize=7.5, color="#94a3b8")
        ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.text(min(proba+0.02, 0.91), 0, f"{proba:.1%}",
                va="center", fontsize=9.5, fontweight="bold", color=clr)
        plt.tight_layout(pad=0.1)
        st.pyplot(fig, use_container_width=True); plt.close()

    with right:
        st.markdown('<div class="card"><div class="card-label">📋 Patient Summary</div>',
                    unsafe_allow_html=True)
        items = list(st.session_state.get("patient_summary", {}).items())
        cc1, cc2 = st.columns(2)
        for i, (k, v) in enumerate(items):
            (cc1 if i%2==0 else cc2).metric(k, v)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div><hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — SHAP
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div id="sec-shap" class="section-anchor">', unsafe_allow_html=True)
st.markdown('<div class="sec-head">🔍 SHAP Explanation <span class="pill">Step 3</span></div>',
            unsafe_allow_html=True)

if not st.session_state.get("show_results"):
    st.markdown('<div class="notice-b">ℹ️ SHAP analysis will appear here after prediction.</div>',
                unsafe_allow_html=True)
else:
    try:
        import shap
        pdf = st.session_state["pdf"]
        with st.spinner("Computing SHAP values…"):
            ex   = shap.TreeExplainer(model) if model else None
            svr  = ex.shap_values(pdf) if ex else None
            sv   = svr[1] if isinstance(svr, list) else svr

        if sv is not None:
            tab1, tab2 = st.tabs(["🌊  Feature Impact", "📊  Global Importance"])

            with tab1:
                ct = (pd.DataFrame({
                    "Feature": pdf.columns,
                    "Value":   pdf.iloc[0].values,
                    "SHAP":    sv[0]
                }).sort_values("SHAP", key=abs, ascending=False)
                  .head(15).reset_index(drop=True))
                ct.index += 1

                fig, ax = clean_fig(9, 5)
                colors = ["#059669" if v>0 else "#dc2626" for v in ct["SHAP"]]
                ax.barh(ct["Feature"], ct["SHAP"],
                        color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
                ax.axvline(0, color="#93c5fd", lw=1.2)
                ax.set_xlabel("SHAP Value — impact on survival probability",
                              color="#374151", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

                def cs(v):
                    return ("background:#d1fae5;color:#065f46" if v>0
                            else "background:#fee2e2;color:#7f1d1d")
                st.markdown("##### Feature Contributions")
                st.dataframe(
                    ct.style.applymap(cs, subset=["SHAP"])
                       .format({"SHAP":"{:.4f}","Value":"{:.3g}"}),
                    use_container_width=True)

            with tab2:
                imp = (pd.DataFrame({
                    "Feature":    pdf.columns,
                    "Mean|SHAP|": np.abs(sv[0])
                }).sort_values("Mean|SHAP|", ascending=True).tail(15))

                fig2, ax2 = clean_fig(9, 5)
                bars = ax2.barh(imp["Feature"], imp["Mean|SHAP|"],
                                color="#3b82f6", alpha=0.82,
                                edgecolor="white", linewidth=0.5)
                # Gradient effect: darker bars = higher importance
                for i, (bar, val) in enumerate(zip(bars, imp["Mean|SHAP|"])):
                    alpha = 0.4 + 0.6 * (i / max(len(bars)-1, 1))
                    bar.set_alpha(alpha)
                ax2.set_xlabel("Mean |SHAP|", color="#374151", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True); plt.close()
        else:
            st.warning("SHAP computation failed — check model compatibility.")
    except ImportError:
        st.warning("Install SHAP: `pip install shap`")
    except Exception as e:
        st.error(f"SHAP error: {e}")

st.markdown('</div><hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div id="sec-metrics" class="section-anchor">', unsafe_allow_html=True)
st.markdown('<div class="sec-head">📊 Model Metrics <span class="pill">Reference</span></div>',
            unsafe_allow_html=True)

df_m = pd.DataFrame({
    "Model":     ["XGBoost ★","LightGBM","Random Forest"],
    "ROC-AUC":   [0.8721, 0.8590, 0.8435],
    "Accuracy":  [0.8289, 0.8158, 0.7895],
    "Precision": [0.8462, 0.8333, 0.8000],
    "Recall":    [0.8148, 0.8148, 0.7778],
    "F1-Score":  [0.8302, 0.8240, 0.7887],
})

st.markdown('<div class="card"><div class="card-label">Model Comparison</div>',
            unsafe_allow_html=True)
st.dataframe(
    df_m.set_index("Model")
        .style.highlight_max(color="#dbeafe", axis=0)
        .format("{:.4f}"),
    use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

kc = st.columns(5)
for col,(v,l) in zip(kc,[
    ("0.8721","ROC-AUC"),("82.9%","Accuracy"),
    ("84.6%","Precision"),("81.5%","Recall"),("83.0%","F1-Score")
]):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                    f'<div class="kpi-l">{l}</div>'
                    f'<div class="kpi-b">★ XGBoost</div></div>',
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
ch, cw = st.columns([1.4, 1])

with ch:
    st.markdown('<div class="card"><div class="card-label">ROC-AUC Comparison</div>',
                unsafe_allow_html=True)
    fig, ax = clean_fig(7, 3.5)
    mdls  = ["XGBoost","LightGBM","Random Forest"]
    sc    = [0.8721, 0.8590, 0.8435]
    clrs  = ["#1d4ed8","#3b82f6","#93c5fd"]
    bars  = ax.barh(mdls, sc, color=clrs, alpha=0.88, height=0.45,
                    edgecolor="white", linewidth=0)
    ax.set_xlim(0.77, 0.91)
    for bar, s in zip(bars, sc):
        ax.text(s+0.001, bar.get_y()+bar.get_height()/2,
                f"{s:.4f}", va="center", fontsize=9, fontweight="700", color="#1e3a5f")
    ax.set_xlabel("ROC-AUC", fontsize=8.5, color="#374151")
    ax.tick_params(left=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with cw:
    st.markdown('<div class="card"><div class="card-label">Why XGBoost?</div>',
                unsafe_allow_html=True)
    st.markdown("""
**ROC-AUC** is the primary metric in medical classification — it measures discrimination
between survivors and non-survivors independent of threshold.

**Recall** is critical: missing a failure (false negative) is far more dangerous
than a false alarm.

XGBoost achieves the best **ROC-AUC (0.8721)** and **Recall (81.5%)**.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Class balance chart
st.markdown('<div class="card"><div class="card-label">⚖️ Class Balance & SMOTE</div>',
            unsafe_allow_html=True)
bi1, bi2 = st.columns([1.8, 1])
with bi1:
    st.markdown("""
**Dataset distribution:** ~60% survived · ~40% not survived

**SMOTE** (Synthetic Minority Oversampling) was applied **only on training data**,
strictly after train/test split, to prevent data leakage into evaluation metrics.
    """)
with bi2:
    fig3, ax3 = plt.subplots(figsize=(3.5, 3.2))
    fig3.patch.set_facecolor("#ffffff")
    ax3.pie([60,40],
            labels=["Survived\n60%","Not Survived\n40%"],
            colors=["#1d4ed8","#dbeafe"],
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize":8.5,"color":"#1e3a5f","fontweight":"600"},
            wedgeprops={"edgecolor":"white","linewidth":2.5})
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=False); plt.close()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # close sec-metrics

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Centrale Casablanca &nbsp;·&nbsp; Coding Week March 2026
  &nbsp;·&nbsp; <strong>Team 19</strong> &nbsp;·&nbsp; k. Zerhouni
</div>""", unsafe_allow_html=True)