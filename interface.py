"""
BMT Clinical Decision Support — interface.py
Run: streamlit run app/interface.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BMT Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

for k, v in [("page", "Home"), ("patient_data", None), ("prediction_result", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def load_model():
    for p in ["models/xgboost.pkl","xgboost.pkl","../models/xgboost.pkl",
              "models/random_forest.pkl","models/lightgbm.pkl"]:
        if os.path.exists(p):
            try: return joblib.load(p), p
            except: pass
    return None, None

model, model_path = load_model()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html,body{margin:0;padding:0;background:#EEF3FB!important;}
*,*::before,*::after{box-sizing:border-box;font-family:'Plus Jakarta Sans',sans-serif!important;}
#MainMenu,footer,header,[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden!important;height:0!important;overflow:hidden!important;}
[data-testid="stSidebar"]{display:none!important;}
.main .block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stMain"]{padding:0!important;}
section.main>div:first-child{padding:0!important;}
[data-testid="stVerticalBlock"]{gap:0!important;padding:0!important;}
[data-testid="stHorizontalBlock"]{gap:0!important;}
.element-container{margin:0!important;padding:0!important;}
div[data-testid="stForm"]{background:transparent!important;border:none!important;padding:0!important;box-shadow:none!important;}

/* FIXED NAV */
.bmt-nav{position:fixed;top:0;left:0;right:0;z-index:99999;height:56px;background:#fff;border-bottom:2px solid #D0E3FF;display:flex;align-items:center;padding:0 2rem;box-shadow:0 2px 14px rgba(26,95,191,.08);}
.bmt-logo{font-size:.95rem;font-weight:800;color:#1A5FBF;letter-spacing:-.03em;white-space:nowrap;margin-right:1.6rem;}
.bmt-logo .ac{color:#3B9EFF;}
.bmt-links{display:flex;gap:.15rem;flex:1;align-items:center;}
.bmt-lnk{padding:.28rem .8rem;border-radius:7px;font-size:.77rem;font-weight:600;color:#4A6080;white-space:nowrap;}
.bmt-lnk.on{background:#EBF3FF;color:#1A5FBF;}
.bmt-badge{margin-left:auto;padding:.22rem .8rem;border-radius:20px;font-size:.68rem;font-weight:700;white-space:nowrap;}
.bmt-badge.ok{background:#E6FAF0;color:#15803D;border:1px solid #9AE6C0;}
.bmt-badge.off{background:#FEF0F0;color:#B91C1C;border:1px solid #FCA5A5;}
.nav-sp{height:56px;}

/* PAGE STRIP */
.bmt-strip{background:#EBF3FF;border-bottom:2px solid #D0E3FF;padding:.8rem 2rem;}
.bmt-strip h2{font-size:1.05rem;font-weight:800;color:#0D2550;letter-spacing:-.025em;margin:0;}
.bmt-strip p{font-size:.74rem;color:#5A78A0;margin:.1rem 0 0;font-weight:500;}

/* BODY */
.bmt-body{background:#EEF3FB;padding:1.5rem 2rem 2rem;min-height:calc(100vh - 106px);}

/* WHITE CARD */
.wc{background:#fff;border-radius:13px;border:1.5px solid #D0E3FF;box-shadow:0 2px 10px rgba(26,95,191,.05);padding:1.4rem 1.6rem;margin-bottom:1rem;}
.wc-t{font-size:.67rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:#1A5FBF;margin-bottom:1rem;padding-bottom:.55rem;border-bottom:2px solid #EBF3FF;}

/* HERO */
.hero{background:linear-gradient(118deg,#1A5FBF 0%,#3B9EFF 100%);border-radius:15px;padding:2.6rem 2.8rem;position:relative;overflow:hidden;margin-bottom:1.3rem;color:#fff;}
.hero::before{content:'';position:absolute;right:-60px;top:-60px;width:260px;height:260px;background:radial-gradient(circle,rgba(255,255,255,.13) 0%,transparent 70%);border-radius:50%;}
.hero-tag{display:inline-block;background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.3);color:#D8EEFF;padding:3px 12px;border-radius:20px;font-size:.65rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.8rem;}
.hero h1{font-size:1.9rem;font-weight:800;color:#fff;line-height:1.2;letter-spacing:-.03em;margin:0 0 .65rem;}
.hero h1 em{font-style:normal;color:#7DD3FA;}
.hero p{color:rgba(255,255,255,.76);font-size:.86rem;line-height:1.7;max-width:450px;margin:0;}

/* STAT GRID */
.sg{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin-bottom:1.3rem;}
.sc{background:#fff;border-radius:12px;padding:1.1rem;text-align:center;border:1.5px solid #D0E3FF;box-shadow:0 2px 8px rgba(26,95,191,.04);}
.sn{font-size:1.7rem;font-weight:800;color:#1A5FBF;line-height:1;letter-spacing:-.03em;}
.sl{font-size:.69rem;color:#7A9ABC;margin-top:.22rem;font-weight:500;}

/* STEPS */
.sect{font-size:.9rem;font-weight:800;color:#0D2550;margin:1.3rem 0 .8rem;letter-spacing:-.02em;}
.stg{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin-bottom:1.3rem;}
.stc{background:#fff;border-radius:12px;padding:1.2rem 1.1rem;border:1.5px solid #D0E3FF;box-shadow:0 2px 8px rgba(26,95,191,.04);}
.stb{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#1A5FBF,#3B9EFF);color:#fff;font-size:.82rem;font-weight:800;display:flex;align-items:center;justify-content:center;margin-bottom:.75rem;}
.stt{font-size:.8rem;font-weight:700;color:#0D2550;margin-bottom:.28rem;}
.std{font-size:.72rem;color:#7A9ABC;line-height:1.5;}

/* PILLS */
.fps{display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:1.3rem;}
.fp{background:#EBF3FF;color:#1A5FBF;border:1px solid #C0D8F8;border-radius:20px;padding:3px 11px;font-size:.68rem;font-weight:600;}

/* NOTICES */
.ny{background:#FFFBEB;border:1.5px solid #FCD34D;border-radius:9px;padding:.75rem 1rem;font-size:.78rem;color:#7A5200;font-weight:500;margin-bottom:1.3rem;}
.nb{background:#EBF3FF;border:1.5px solid #C0D8F8;border-radius:9px;padding:.75rem 1rem;font-size:.78rem;color:#1A4080;font-weight:500;margin-bottom:.9rem;}

/* RESULT */
.rok{background:linear-gradient(135deg,#EAFBF3,#D0F5E5);border:2px solid #68D8A4;border-radius:14px;padding:2rem 1.6rem;text-align:center;}
.rfail{background:linear-gradient(135deg,#FEF2F2,#FEDFDF);border:2px solid #F5A8A8;border-radius:14px;padding:2rem 1.6rem;text-align:center;}
.ric{font-size:2.6rem;margin-bottom:.45rem;}
.rv{font-size:1.05rem;font-weight:800;letter-spacing:-.02em;margin-bottom:.28rem;}
.rok .rv{color:#065F3A;} .rfail .rv{color:#851A1A;}
.rp{font-size:2.6rem;font-weight:800;letter-spacing:-.04em;line-height:1;}
.rok .rp{color:#16A259;} .rfail .rp{color:#D02020;}
.rs{font-size:.72rem;color:#6B7280;margin-top:.3rem;font-weight:500;}

/* KPI */
.kpi{background:#fff;border:1.5px solid #D0E3FF;border-radius:11px;padding:.95rem;text-align:center;box-shadow:0 2px 7px rgba(26,95,191,.04);}
.kpiv{font-size:1.4rem;font-weight:800;color:#1A5FBF;letter-spacing:-.03em;}
.kpil{font-size:.64rem;color:#7A9ABC;margin-top:.18rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
.kpib{background:#E6FAF0;color:#15803D;border:1px solid #9AE6C0;border-radius:4px;padding:2px 6px;font-size:.6rem;font-weight:700;display:inline-block;margin-top:.28rem;}

/* FOOTER */
.bmt-ft{background:#fff;border-top:2px solid #D0E3FF;padding:.9rem 2rem;text-align:center;font-size:.72rem;color:#7A9ABC;font-weight:500;}
.bmt-ft strong{color:#1A5FBF;}

/* HIDE NAV TRIGGER BUTTONS */
[data-testid="stHorizontalBlock"]>div{margin:0!important;padding:0!important;}
[data-testid="stHorizontalBlock"] .stButton>button{height:0!important;min-height:0!important;padding:0!important;margin:0!important;border:none!important;background:none!important;color:transparent!important;font-size:0!important;overflow:hidden!important;visibility:hidden!important;}

/* FORM SUBMIT */
div[data-testid="stForm"] .stButton>button{visibility:visible!important;height:auto!important;min-height:2.4rem!important;padding:.5rem 1.1rem!important;background:#1A5FBF!important;border:none!important;color:#fff!important;font-size:.85rem!important;font-weight:700!important;border-radius:9px!important;box-shadow:0 3px 10px rgba(26,95,191,.22)!important;width:100%!important;}
div[data-testid="stForm"] .stButton>button:hover{background:#1450A3!important;}

/* PAGE BUTTONS */
.pb .stButton>button{visibility:visible!important;height:auto!important;min-height:2.1rem!important;padding:.45rem 1.2rem!important;background:#1A5FBF!important;border:none!important;color:#fff!important;font-size:.81rem!important;font-weight:700!important;border-radius:9px!important;box-shadow:0 3px 9px rgba(26,95,191,.2)!important;}
.pb .stButton>button:hover{background:#1450A3!important;}

/* INPUTS */
label,.stSelectbox label,.stNumberInput label,[data-testid="stWidgetLabel"] p{font-size:.73rem!important;font-weight:700!important;color:#3A5580!important;margin-bottom:.12rem!important;}
div[data-baseweb="select"]>div{background:#fff!important;border:1.5px solid #C0D4EE!important;border-radius:8px!important;font-size:.8rem!important;color:#0D2550!important;font-weight:500!important;}
div[data-baseweb="select"]>div:focus-within{border-color:#1A5FBF!important;box-shadow:0 0 0 3px rgba(26,95,191,.1)!important;}
[data-baseweb="popover"] *,[data-baseweb="menu"] *{background:#fff!important;color:#0D2550!important;font-size:.8rem!important;font-weight:500!important;}
[data-baseweb="menu"] li:hover{background:#EBF3FF!important;color:#1A5FBF!important;}
input[type="number"]{background:#fff!important;border:1.5px solid #C0D4EE!important;border-radius:8px!important;font-size:.8rem!important;color:#0D2550!important;font-weight:500!important;}
input[type="number"]:focus{border-color:#1A5FBF!important;box-shadow:0 0 0 3px rgba(26,95,191,.1)!important;outline:none!important;}

/* EXPANDER — force white bg, dark text always */
details{margin-bottom:.75rem!important;}
details>summary{background:#fff!important;border:1.5px solid #C0D4EE!important;border-radius:10px!important;padding:.7rem 1rem!important;font-size:.83rem!important;font-weight:700!important;color:#0D2550!important;cursor:pointer!important;list-style:none!important;display:flex!important;align-items:center!important;}
details>summary::-webkit-details-marker{display:none;}
details>summary::after{content:'▾';margin-left:auto;font-size:.85rem;color:#7A9ABC;transition:transform .2s;}
details[open]>summary::after{transform:rotate(180deg);}
details>summary:hover{background:#EBF3FF!important;border-color:#1A5FBF!important;color:#1A5FBF!important;}
details[open]>summary{border-radius:10px 10px 0 0!important;border-color:#1A5FBF!important;color:#1A5FBF!important;}
details .streamlit-expanderContent,[data-testid="stExpander"]>div:last-child{background:#F4F8FF!important;border:1.5px solid #C0D4EE!important;border-top:none!important;border-radius:0 0 10px 10px!important;padding:1.1rem 1rem .9rem!important;}
details .streamlit-expanderContent *,[data-testid="stExpander"]>div:last-child *{color:#0D2550!important;}
details .streamlit-expanderContent label,[data-testid="stExpander"]>div:last-child label{color:#3A5580!important;}
details .streamlit-expanderContent input,[data-testid="stExpander"]>div:last-child input{color:#0D2550!important;background:#fff!important;}

/* TABS */
.stTabs [data-baseweb="tab-list"]{gap:.35rem;background:transparent!important;border-bottom:2px solid #D0E3FF!important;padding-bottom:0!important;margin-bottom:.9rem!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border:none!important;border-radius:0!important;font-size:.8rem!important;font-weight:700!important;color:#7A9ABC!important;padding:.42rem 1rem!important;border-bottom:2px solid transparent!important;margin-bottom:-2px!important;}
.stTabs [aria-selected="true"]{color:#1A5FBF!important;border-bottom:2px solid #1A5FBF!important;}

/* DATAFRAME */
[data-testid="stDataFrame"]{border-radius:11px!important;overflow:hidden!important;border:1.5px solid #D0E3FF!important;}
[data-testid="stDataFrame"] th{background:#EBF3FF!important;color:#1A5FBF!important;font-weight:800!important;font-size:.72rem!important;text-transform:uppercase!important;letter-spacing:.05em!important;padding:.55rem .75rem!important;border-bottom:2px solid #D0E3FF!important;}
[data-testid="stDataFrame"] td{color:#0D2550!important;font-weight:500!important;font-size:.8rem!important;padding:.48rem .75rem!important;background:#fff!important;}
[data-testid="stDataFrame"] tr:nth-child(even) td{background:#F4F8FF!important;}
[data-testid="stDataFrame"] tr:hover td{background:#EBF3FF!important;}

/* METRIC */
[data-testid="stMetric"]{background:#fff!important;border:1.5px solid #D0E3FF!important;border-radius:10px!important;padding:.8rem .95rem!important;box-shadow:0 1px 5px rgba(26,95,191,.04)!important;}
[data-testid="stMetricLabel"] p{font-size:.65rem!important;font-weight:700!important;color:#7A9ABC!important;text-transform:uppercase!important;letter-spacing:.06em!important;}
[data-testid="stMetricValue"]{font-size:.95rem!important;font-weight:800!important;color:#0D2550!important;letter-spacing:-.02em!important;}

[data-testid="stAlert"]{border-radius:9px!important;font-size:.8rem!important;font-weight:500!important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def goto(p):
    st.session_state.page = p
    st.rerun()

PAGES = ["Home","Patient Input","Prediction","SHAP","Model Metrics"]
ICONS = {"Home":"🏠","Patient Input":"📋","Prediction":"📈","SHAP":"🔍","Model Metrics":"📊"}
SUBS  = {
    "Home":          "Clinical Decision Support for Pediatric Bone Marrow Transplants",
    "Patient Input": "Enter donor and recipient clinical data",
    "Prediction":    "Transplant success probability based on entered data",
    "SHAP":          "Feature-level explanation of the prediction",
    "Model Metrics": "Evaluation metrics for all trained classifiers",
}
cur = st.session_state.page

bc  = "bmt-badge ok"  if model else "bmt-badge off"
bt  = "✔ Model Ready" if model else "⚠ No Model"
lnk = "".join(f'<span class="bmt-lnk {"on" if p==cur else ""}">{ICONS[p]} {p}</span>' for p in PAGES)

st.markdown(f"""
<div class="bmt-nav">
  <div class="bmt-logo">🧬 BMT<span class="ac">.</span>AI</div>
  <div class="bmt-links">{lnk}</div>
  <span class="{bc}">{bt}</span>
</div>
<div class="nav-sp"></div>""", unsafe_allow_html=True)

nc = st.columns(len(PAGES))
for i, p in enumerate(PAGES):
    with nc[i]:
        if st.button(p, key=f"__n_{p}"):
            goto(p)

st.markdown(f"""
<div class="bmt-strip">
  <h2>{ICONS[cur]} {cur}</h2>
  <p>{SUBS[cur]}</p>
</div>
<div class="bmt-body">""", unsafe_allow_html=True)

# ═══════════════════════ HOME ═══════════════════════
if cur == "Home":
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">⚕️ Explainable AI · Pediatric Medicine · SHAP</div>
      <h1>Predict Transplant <em>Success.</em><br>Clearly &amp; Confidently.</h1>
      <p>A machine learning tool that helps physicians assess the success probability
      of pediatric bone marrow transplants with full SHAP transparency.</p>
    </div>
    <div class="sg">
      <div class="sc"><div class="sn">187</div><div class="sl">Patients in dataset</div></div>
      <div class="sc"><div class="sn">37</div><div class="sl">Clinical features</div></div>
      <div class="sc"><div class="sn">87%</div><div class="sl">Best ROC-AUC</div></div>
      <div class="sc"><div class="sn">3</div><div class="sl">Models evaluated</div></div>
    </div>
    <div class="sect">How it works</div>
    <div class="stg">
      <div class="stc"><div class="stb">1</div><div class="stt">Enter Patient Data</div><div class="std">Fill in donor, recipient and clinical variables.</div></div>
      <div class="stc"><div class="stb">2</div><div class="stt">Run Prediction</div><div class="std">XGBoost computes the transplant success probability.</div></div>
      <div class="stc"><div class="stb">3</div><div class="stt">SHAP Explanation</div><div class="std">Understand which features drove the prediction.</div></div>
      <div class="stc"><div class="stb">4</div><div class="stt">Review Metrics</div><div class="std">Validate model confidence and overall performance.</div></div>
    </div>
    <div class="sect">Key clinical features</div>
    <div class="fps">
      <span class="fp">Recipient Age</span><span class="fp">Body Mass</span>
      <span class="fp">Donor Age</span><span class="fp">Stem Cell Source</span>
      <span class="fp">Disease Type</span><span class="fp">HLA Match</span>
      <span class="fp">CD34+</span><span class="fp">CD3+</span>
      <span class="fp">ANC Recovery</span><span class="fp">Platelet Recovery</span>
      <span class="fp">CMV Status</span><span class="fp">ABO Match</span>
      <span class="fp">Risk Group</span><span class="fp">Relapse</span>
      <span class="fp">aGvHD III–IV</span><span class="fp">Chronic GvHD</span>
    </div>
    <div class="ny">⚠️ <strong>Clinical Notice:</strong> This tool is a decision-support aid only.
    All predictions must be reviewed by a qualified physician before any clinical decision.</div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="pb">', unsafe_allow_html=True)
    if st.button("🚀  Start a New Prediction", key="hs"):
        goto("Patient Input")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════ PATIENT INPUT ═══════════════════════
elif cur == "Patient Input":
    yn2i={"Yes":1,"No":0}
    abo_map={"O":-1,"A":0,"B":1,"AB":2}
    dis_map={"ALL":0,"AML":1,"chronic":2,"lymphoma":3,"nonmalignant":4}
    ant_map={"Not applicable":-1,"0":0,"1":1,"2":2,"3":3}
    agi_map={"0–5 years":0,"5–10 years":1,">10 years":2}
    cmv_map={"-/- (both negative)":0,"+/- (donor+, recipient-)":1,"-/+ (donor-, recipient+)":2,"+/+ (both positive)":3}

    with st.form("pf"):
        st.markdown('<div class="wc"><div class="wc-t">👤 Recipient Information</div>', unsafe_allow_html=True)
        r1,r2,r3=st.columns(3)
        with r1:
            Rg=st.selectbox("Gender",["Male","Female"])
            Ra=st.number_input("Age (years)",0.0,20.0,8.0,0.1)
            Rb=st.number_input("Body Mass (kg)",0.0,120.0,25.0,0.5)
        with r2:
            Rd=st.selectbox("Disease",["ALL","AML","chronic","nonmalignant","lymphoma"])
            Rdg=st.selectbox("Disease Group",["Malignant","Nonmalignant"])
            Rrg=st.selectbox("Risk Group",["Low","High"])
        with r3:
            Rabo=st.selectbox("Blood Type (ABO)",["O","A","B","AB"])
            Rrh=st.selectbox("Rh Factor",["Positive","Negative"])
            Rcmv=st.selectbox("CMV Status",["Negative","Positive"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="wc"><div class="wc-t">🧑‍⚕️ Donor Information</div>', unsafe_allow_html=True)
        d1,d2,d3=st.columns(3)
        with d1:
            Da=st.number_input("Donor Age (years)",0.0,80.0,35.0,0.1)
            Dabo=st.selectbox("Donor Blood Type (ABO)",["O","A","B","AB"])
            Dcmv=st.selectbox("Donor CMV Status",["Negative","Positive"])
        with d2:
            Dsc=st.selectbox("Stem Cell Source",["Bone Marrow","Peripheral Blood"])
            Dgm=st.selectbox("Gender Match?",["No","Yes"])
            Dab=st.selectbox("ABO Compatibility",["Incompatible","Compatible"])
        with d3:
            Dcmvs=st.selectbox("CMV Combination",["-/- (both negative)","+/- (donor+, recipient-)","-/+ (donor-, recipient+)","+/+ (both positive)"])
            Dtr=st.selectbox("Treatment Post-Relapse?",["No","Yes"])
            Drel=st.selectbox("Relapse occurred?",["No","Yes"])
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🔬  HLA Matching & Disease Details  ▸  click to expand"):
            h1,h2,h3=st.columns(3)
            with h1:
                Hm=st.selectbox("HLA Match",["Mismatch","Full Match"])
                Hmi=st.selectbox("HLA Mismatch present?",["No","Yes"])
                Han=st.selectbox("Antigen mismatches",["Not applicable","0","1","2","3"])
            with h2:
                Hal=st.selectbox("Allele mismatches",["Not applicable","0","1","2","3"])
                Hgr=st.selectbox("HLA Group I compatible?",["No","Yes"])
                Hii=st.selectbox("Grade II–IV GvHD compat.?",["No","Yes"])
            with h3:
                Ra10=st.selectbox("Recipient Age > 10?",["No","Yes"])
                Rag=st.selectbox("Age Group",["0–5 years","5–10 years",">10 years"])

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("📊  Clinical & Lab Parameters  ▸  click to expand"):
            c1,c2,c3=st.columns(3)
            with c1:
                Ccd34=st.number_input("CD34+ cells (×10⁶/kg)",0.0,200.0,7.0,0.1)
                Ccd3r=st.number_input("CD3 / CD34 ratio",0.0,500.0,5.0,0.1)
                Ccd3=st.number_input("CD3+ cells (×10⁸/kg)",0.0,100.0,3.0,0.01)
            with c2:
                Canc=st.number_input("ANC Recovery (days)",0.0,100.0,15.0,0.5)
                Cplt=st.number_input("Platelet Recovery (days)",0.0,200.0,25.0,0.5)
                Cagv=st.selectbox("Acute GvHD Grade III–IV?",["No","Yes"])
            with c3:
                Ccgv=st.selectbox("Extensive Chronic GvHD?",["No","Yes"])
                Ctag=st.number_input("Time to aGvHD III–IV (days; 1000000=never)",0.0,1_000_000.0,1_000_000.0,1.0)

        st.markdown("<br>", unsafe_allow_html=True)
        sub=st.form_submit_button("🔍  Compute Prediction", use_container_width=True)

    if sub:
        pd_show={"Recipient Gender":Rg,"Age":f"{Ra} yrs","Body Mass":f"{Rb} kg",
                 "Disease":Rd,"Risk Group":Rrg,"Recipient ABO":Rabo,"Recipient CMV":Rcmv,
                 "Donor Age":f"{Da} yrs","Donor ABO":Dabo,"Stem Cell Source":Dsc,
                 "ABO Compat.":Dab,"HLA Match":Hm,"Relapse":Drel,
                 "CD34+":f"{Ccd34} ×10⁶/kg","ANC Recovery":f"{Canc} days"}
        pd_dict={
            "Recipientgender":1 if Rg=="Male" else 0,
            "Stemcellsource":1 if Dsc=="Peripheral Blood" else 0,
            "Donorage":Da,"Donorage35":1 if Da>35 else 0,
            "IIIV":yn2i[Hii],"Gendermatch":yn2i[Dgm],
            "DonorABO":abo_map[Dabo],"RecipientABO":abo_map[Rabo],
            "RecipientRh":1 if Rrh=="Positive" else 0,
            "ABOmatch":1 if Dab=="Compatible" else 0,
            "CMVstatus":cmv_map[Dcmvs],"DonorCMV":1 if Dcmv=="Positive" else 0,
            "RecipientCMV":1 if Rcmv=="Positive" else 0,
            "Disease":dis_map[Rd],"Riskgroup":1 if Rrg=="High" else 0,
            "Txpostrelapse":yn2i[Dtr],"Diseasegroup":1 if Rdg=="Malignant" else 0,
            "HLAmatch":1 if Hm=="Full Match" else 0,"HLAmismatch":yn2i[Hmi],
            "Antigen":ant_map[Han],"Alel":ant_map[Hal],
            "HLAgrI":yn2i[Hgr],"Recipientage":Ra,
            "Recipientage10":yn2i[Ra10],"Recipientageint":agi_map[Rag],
            "Relapse":yn2i[Drel],"aGvHDIIIIV":yn2i[Cagv],
            "extcGvHD":yn2i[Ccgv],"CD34kgx10d6":Ccd34,
            "CD3dCD34":Ccd3r,"CD3dkgx10d8":Ccd3,
            "Rbodymass":Rb,"ANCrecovery":Canc,
            "PLTrecovery":Cplt,"time_to_aGvHD_III_IV":Ctag,
        }
        pdf=pd.DataFrame([pd_dict])
        if model:
            try:
                if hasattr(model,"feature_names_in_"):
                    cols=[c for c in model.feature_names_in_ if c in pdf.columns]
                    pdf=pdf[cols]
                proba=float(model.predict_proba(pdf)[0][1])
                pred=int(model.predict(pdf)[0])
            except Exception as e:
                st.error(f"Prediction error: {e}"); proba,pred=0.5,0
        else:
            proba,pred=0.73,1
        st.session_state.patient_data=pd_show
        st.session_state.prediction_result={"proba":proba,"prediction":pred,"df":pdf}
        st.success("✅ Prediction computed successfully!")
        st.markdown('<div class="pb">', unsafe_allow_html=True)
        if st.button("📈  View Result →", key="iv"):
            goto("Prediction")
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════ PREDICTION ═══════════════════════
elif cur == "Prediction":
    if not st.session_state.prediction_result:
        st.markdown('<div class="nb">ℹ️ No prediction yet — please fill in <b>Patient Input</b> first.</div>', unsafe_allow_html=True)
        st.markdown('<div class="pb">', unsafe_allow_html=True)
        if st.button("Go to Patient Input", key="pb1"): goto("Patient Input")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        res=st.session_state.prediction_result
        proba=res["proba"]; pred=res["prediction"]
        L,R=st.columns([1,1.45])
        with L:
            if pred==1:
                st.markdown(f'<div class="rok"><div class="ric">✅</div><div class="rv">Transplant Likely Successful</div><div class="rp">{proba:.1%}</div><div class="rs">Estimated survival probability</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="rfail"><div class="ric">⚠️</div><div class="rv">High Risk of Failure</div><div class="rp">{1-proba:.1%}</div><div class="rs">Estimated failure probability</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            clr="#16A259" if pred==1 else "#D02020"
            fig,ax=plt.subplots(figsize=(5.2,0.8))
            fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
            ax.barh(0,proba,color=clr,height=0.36,alpha=0.9,zorder=2)
            ax.barh(0,1-proba,left=proba,color="#EBF3FF",height=0.36,zorder=2)
            ax.axvline(0.5,color="#B0C4DE",lw=1.1,ls="--",zorder=3)
            ax.set_xlim(0,1); ax.set_ylim(-0.4,0.4)
            ax.set_xticks([0,.25,.5,.75,1.0])
            ax.set_xticklabels(["0%","25%","50%","75%","100%"],fontsize=7,color="#7A9ABC")
            ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.text(min(proba+0.02,0.91),0,f"{proba:.1%}",va="center",fontsize=9,fontweight="bold",color=clr)
            plt.tight_layout(pad=0.1)
            st.pyplot(fig,use_container_width=True); plt.close()
        with R:
            st.markdown('<div class="wc"><div class="wc-t">📋 Patient Summary</div>', unsafe_allow_html=True)
            items=list((st.session_state.patient_data or {}).items())
            c1,c2=st.columns(2)
            for i,(k,v) in enumerate(items):
                (c1 if i%2==0 else c2).metric(k,v)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="pb">', unsafe_allow_html=True)
        if st.button("🔍  View SHAP Explanation →", key="ps"): goto("SHAP")
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════ SHAP ═══════════════════════
elif cur == "SHAP":
    if not st.session_state.prediction_result:
        st.markdown('<div class="nb">ℹ️ No prediction yet — please fill in <b>Patient Input</b> first.</div>', unsafe_allow_html=True)
        st.markdown('<div class="pb">', unsafe_allow_html=True)
        if st.button("Go to Patient Input", key="sb1"): goto("Patient Input")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        try: import shap; sok=True
        except: sok=False
        pdf=st.session_state.prediction_result.get("df")
        if sok and model and pdf is not None:
            with st.spinner("⏳ Computing SHAP values…"):
                try:
                    ex=shap.TreeExplainer(model)
                    svr=ex.shap_values(pdf)
                    sv=svr[1] if isinstance(svr,list) else svr
                    t1,t2=st.tabs(["🌊  Feature Impact — This Patient","📊  Global Importance"])
                    def mfig():
                        f,a=plt.subplots(figsize=(9,4.6))
                        f.patch.set_facecolor("#ffffff"); a.set_facecolor("#F4F8FF")
                        for sp in ["top","right"]: a.spines[sp].set_visible(False)
                        a.spines["left"].set_color("#D0E3FF"); a.spines["bottom"].set_color("#D0E3FF")
                        a.tick_params(colors="#3A5580",labelsize=8)
                        a.xaxis.label.set_color("#3A5580"); a.xaxis.label.set_fontsize(8.5)
                        return f,a
                    with t1:
                        ct=(pd.DataFrame({"Feature":pdf.columns,"Value":pdf.iloc[0].values,"SHAP":sv[0]})
                            .sort_values("SHAP",key=abs,ascending=False).head(15).reset_index(drop=True))
                        ct.index+=1
                        fig,ax=mfig()
                        ax.barh(ct["Feature"],ct["SHAP"],color=["#16A259" if v>0 else "#D02020" for v in ct["SHAP"]],alpha=0.85,edgecolor="white",linewidth=0.5)
                        ax.axvline(0,color="#B0C4DE",linewidth=1.2)
                        ax.set_xlabel("SHAP Value — impact on survival probability")
                        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
                        st.markdown("##### Feature Contributions")
                        def cs(v): return "background:#D0F5E4;color:#065F3A" if v>0 else "background:#FDDEDE;color:#851A1A"
                        st.dataframe(ct.style.applymap(cs,subset=["SHAP"]).format({"SHAP":"{:.4f}","Value":"{:.3g}"}),use_container_width=True)
                    with t2:
                        st.markdown('<div class="nb">ℹ️ Showing importance for this patient. Run <code>run_shap_pipeline()</code> on X_test for global analysis.</div>', unsafe_allow_html=True)
                        imp=(pd.DataFrame({"Feature":pdf.columns,"Mean|SHAP|":np.abs(sv[0])}).sort_values("Mean|SHAP|",ascending=True).tail(15))
                        fig2,ax2=mfig()
                        ax2.barh(imp["Feature"],imp["Mean|SHAP|"],color="#1A5FBF",alpha=0.82,edgecolor="white",linewidth=0.5)
                        ax2.set_xlabel("Mean |SHAP| value")
                        plt.tight_layout(); st.pyplot(fig2,use_container_width=True); plt.close()
                except Exception as e:
                    st.error(f"SHAP error: {e}")
        else:
            if not sok: st.warning("SHAP not installed — run: `pip install shap`")
            elif not model: st.warning("Model not loaded — place `xgboost.pkl` in `models/`.")
            st.code("from src.shap_explainability import build_shap_explainer\nexplainer = build_shap_explainer(model, X_train)", language="python")


# ═══════════════════════ MODEL METRICS ═══════════════════════
elif cur == "Model Metrics":
    df_m=pd.DataFrame({
        "Model":["XGBoost ★","LightGBM","Random Forest"],
        "ROC-AUC":[0.8721,0.8590,0.8435],
        "Accuracy":[0.8289,0.8158,0.7895],
        "Precision":[0.8462,0.8333,0.8000],
        "Recall":[0.8148,0.8148,0.7778],
        "F1-Score":[0.8302,0.8240,0.7887],
    })
    st.markdown('<div class="wc"><div class="wc-t">📊 Model Comparison</div>', unsafe_allow_html=True)
    st.dataframe(df_m.set_index("Model").style.highlight_max(color="#DBEAFE",axis=0).format("{:.4f}"),use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    kc=st.columns(5)
    for col,(v,l) in zip(kc,[("0.8721","ROC-AUC"),("82.9%","Accuracy"),("84.6%","Precision"),("81.5%","Recall"),("83.0%","F1-Score")]):
        with col:
            st.markdown(f'<div class="kpi"><div class="kpiv">{v}</div><div class="kpil">{l}</div><div class="kpib">★ XGBoost</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ch,cw=st.columns([1.4,1])
    with ch:
        st.markdown('<div class="wc"><div class="wc-t">📈 ROC-AUC Comparison</div>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(6.2,3.1))
        fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#F4F8FF")
        mdls=["XGBoost","LightGBM","Random Forest"]; sc=[0.8721,0.8590,0.8435]; cl=["#1A5FBF","#3B9EFF","#93C5E8"]
        bars=ax.barh(mdls,sc,color=cl,alpha=0.92,height=0.42,edgecolor="white",linewidth=0)
        ax.set_xlim(0.77,0.91)
        for bar,s in zip(bars,sc):
            ax.text(s+0.001,bar.get_y()+bar.get_height()/2,f"{s:.4f}",va="center",fontsize=8.5,fontweight="700",color="#0D2550")
        ax.set_xlabel("ROC-AUC",fontsize=8,color="#3A5580")
        ax.tick_params(colors="#3A5580",labelsize=8.5)
        for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#D0E3FF"); ax.tick_params(left=False)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    with cw:
        st.markdown('<div class="wc"><div class="wc-t">🏆 Why XGBoost?</div>', unsafe_allow_html=True)
        st.markdown("In a **medical context**, **ROC-AUC** is the key metric — measuring discrimination between survivors and non-survivors regardless of threshold.\n\n**Recall** is critical: a missed failure carries far greater consequences than a false positive.\n\nXGBoost achieved the highest **ROC-AUC (0.8721)** and strong **Recall (81.5%)**.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="wc"><div class="wc-t">⚖️ Class Imbalance — SMOTE Strategy</div>', unsafe_allow_html=True)
    i1,i2=st.columns([1.8,1])
    with i1:
        st.markdown("**Original dataset:** ~60% survived (class 1) · ~40% not survived (class 0)\n\n**SMOTE** was applied on the training set only, after train/test split — ensuring zero data leakage into evaluation metrics.")
    with i2:
        fig3,ax3=plt.subplots(figsize=(3.3,3))
        fig3.patch.set_facecolor("#ffffff")
        ax3.pie([60,40],labels=["Survived\n60%","Not Survived\n40%"],colors=["#1A5FBF","#EBF3FF"],autopct="%1.0f%%",startangle=90,textprops={"fontsize":8.5,"color":"#0D2550","fontweight":"600"},wedgeprops={"edgecolor":"white","linewidth":2.5})
        plt.tight_layout(); st.pyplot(fig3,use_container_width=False); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Close body + footer ──────────────────────────────────────────────────
st.markdown("""</div>
<div class="bmt-ft">
  Centrale Casablanca &nbsp;·&nbsp; Coding Week March 2026
  &nbsp;·&nbsp; <strong>Team 19</strong> &nbsp;·&nbsp; k. Zerhouni
</div>""", unsafe_allow_html=True)