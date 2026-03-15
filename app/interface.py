"""
Bone Marrow Transplant Decision Support System
Centrale Casablanca — Coding Week Project 4, March 2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, sqlite3, hashlib, json, warnings, joblib
warnings.filterwarnings("ignore")
import streamlit.components.v1 as components

st.set_page_config(page_title="BMT Decision Support", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# ═══ MEGA CSS — Blue Medical Theme with Toggles & Sliders ════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');

:root {
    --blue-deep: #0a1628;
    --blue-mid: #1a3a5c;
    --blue-light: #3b82f6;
    --blue-glow: #60a5fa;
    --blue-soft: #dbeafe;
    --purple-accent: #a78bfa;
    --pink-accent: #ec4899;
    --glass-bg: rgba(255,255,255,0.08);
    --glass-border: rgba(255,255,255,0.12);
    --text-white: #f0f4ff;
    --text-muted: #94a3b8;
    --surface: rgba(15,25,50,0.85);
    --green: #22c55e;
    --red: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #142644 30%, #1a3a5c 60%, #2563a8 100%) !important;
    color: var(--text-white) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

/* All text white */
p, span, label, .stMarkdown, [data-testid="stWidgetLabel"] p,
div[data-testid="stText"], .element-container {
    color: var(--text-white) !important;
}
h1, h2, h3, h4 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-white) !important;
    font-weight: 700 !important;
}

/* Sidebar — dark blue glass */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060e1f 0%, #0d1b35 40%, #142644 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.15) !important;
}
section[data-testid="stSidebar"] * { color: #c7d2fe !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(59,130,246,0.12) !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(59,130,246,0.1) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    color: #c7d2fe !important;
    text-align: left !important;
    padding: 10px 16px !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(59,130,246,0.2) !important;
    border-color: rgba(59,130,246,0.4) !important;
}

/* Glass cards */
.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 16px;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, rgba(59,130,246,0.15) 0%, rgba(167,139,250,0.1) 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 22px;
    padding: 38px 42px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -40px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { color: #f0f4ff !important; font-size: 28px !important; margin-bottom: 4px !important; }
.hero p { color: rgba(148,163,184,0.9); font-size: 14px; margin: 0; }
.hero .gp {
    display: inline-block;
    background: rgba(167,139,250,0.2);
    border: 1px solid rgba(167,139,250,0.35);
    color: #c4b5fd;
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Metric cards */
.mrow { display: flex; gap: 12px; margin-bottom: 22px; flex-wrap: wrap; }
.mc {
    flex: 1; min-width: 120px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px 18px;
    text-align: center;
}
.mc .ml { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 5px; }
.mc .mv { font-size: 24px; font-weight: 800; font-family: 'Fira Code', monospace; color: var(--text-white); }

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 14px;
    color: var(--text-white);
}

/* Prediction results */
.pg {
    background: linear-gradient(135deg, rgba(34,197,94,0.15) 0%, rgba(34,197,94,0.05) 100%);
    border: 2px solid rgba(34,197,94,0.5);
    border-radius: 20px; padding: 34px;
    text-align: center; margin: 22px 0;
}
.pg h2 { color: #86efac !important; font-size: 23px !important; }

.pb {
    background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(239,68,68,0.05) 100%);
    border: 2px solid rgba(239,68,68,0.5);
    border-radius: 20px; padding: 34px;
    text-align: center; margin: 22px 0;
}
.pb h2 { color: #fca5a5 !important; font-size: 23px !important; }

.bar { background: rgba(255,255,255,0.1); border-radius: 10px; height: 18px; margin: 12px auto; overflow: hidden; max-width: 360px; }
.bf { height: 100%; border-radius: 10px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 11px 28px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 25px rgba(59,130,246,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Warning box */
.wb {
    background: rgba(234,179,8,0.1);
    border: 1px solid rgba(234,179,8,0.3);
    border-radius: 12px; padding: 14px 18px; margin-top: 18px;
    color: #fde68a;
}

/* Pill */
.pill { display: inline-block; padding: 3px 12px; border-radius: 14px; font-size: 11px; font-weight: 600; }
.pill-g { background: rgba(34,197,94,0.15); color: #86efac; border: 1px solid rgba(34,197,94,0.3); }

/* Expander */
.streamlit-expanderHeader { font-weight: 600 !important; font-size: 14px !important; color: var(--text-white) !important; }
details { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 14px !important; }

/* Input styling */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: var(--text-white) !important;
}
[data-baseweb="select"] > div { background: rgba(255,255,255,0.06) !important; }
[data-baseweb="menu"] { background: rgba(15,25,50,0.95) !important; }
[data-baseweb="menu"] li { color: var(--text-white) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid rgba(59,130,246,0.15) !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; font-weight: 500 !important; background: transparent !important; }
.stTabs [aria-selected="true"] { color: #60a5fa !important; border-bottom: 2px solid #60a5fa !important; }

/* Toggle switch styling for selectbox Yes/No - override via CSS */
.stSelectbox label { color: #94a3b8 !important; font-size: 13px !important; font-weight: 600 !important; }
.stNumberInput label { color: #94a3b8 !important; font-size: 13px !important; font-weight: 600 !important; }

/* Slider styling - modern circle on line */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    height: 4px !important;
    border-radius: 4px !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: white !important;
    border: 3px solid #3b82f6 !important;
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.4) !important;
}

/* Alerts */
.stAlert { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; }
.stAlert p { color: var(--text-white) !important; }

/* Dataframe */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }

/* Footer */
.bmt-ft {
    margin-top: 60px;
    padding: 40px 0 30px;
    border-top: 1px solid rgba(59,130,246,0.15);
    text-align: center;
}
.ft-in { max-width: 600px; margin: 0 auto; }
.ft-logo { font-size: 18px; font-weight: 800; color: #60a5fa; letter-spacing: 0.5px; margin-bottom: 12px; }
.ft-div { width: 60px; height: 2px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); margin: 12px auto; border-radius: 2px; }
.ft-main { font-size: 13px; color: #94a3b8; margin-bottom: 8px; line-height: 1.6; }
.ft-main strong { color: #c7d2fe; }
.ft-credit { font-size: 12px; color: #64748b; margin-top: 8px; }
.ft-credit strong { color: #94a3b8; }

/* Welcome page */
.welcome-wrap {
    min-height: 90vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    position: relative;
}
.welcome-title {
    font-size: 48px;
    font-weight: 800;
    color: white;
    letter-spacing: -0.02em;
    margin-bottom: 16px;
    text-shadow: 0 4px 30px rgba(59,130,246,0.3);
}
.welcome-title span { color: #60a5fa; }
.welcome-sub {
    font-size: 18px;
    color: #94a3b8;
    max-width: 500px;
    line-height: 1.6;
    margin-bottom: 40px;
}
.welcome-features {
    display: flex; gap: 24px; margin-bottom: 48px; flex-wrap: wrap; justify-content: center;
}
.wf-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px 28px;
    width: 180px;
    text-align: center;
}
.wf-card .wf-icon { font-size: 32px; margin-bottom: 10px; }
.wf-card .wf-label { font-size: 13px; font-weight: 600; color: #c7d2fe; }

/* Toggle switch for Yes/No fields */
.toggle-row {
    display: flex; align-items: center; gap: 12px; margin: 8px 0;
}
.toggle-label { font-size: 13px; font-weight: 600; color: #94a3b8; min-width: 160px; }
.toggle-val { font-size: 12px; font-weight: 600; color: #60a5fa; }
</style>""", unsafe_allow_html=True)


_HERE = os.path.dirname(os.path.abspath(__file__))

# ═══ DATABASE (UNTOUCHED) ═════════════════════════════════════════════════
DB = os.path.join(_HERE, "bmt_app.db")
def init_db():
    cn=sqlite3.connect(DB);c=cn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,username TEXT UNIQUE,pw TEXT,name TEXT,role TEXT DEFAULT 'physician')")
    c.execute("CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER,patient_json TEXT,prediction INTEGER,probability REAL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    if c.execute("SELECT COUNT(*) FROM users").fetchone()[0]==0:
        c.execute("INSERT INTO users(username,pw,name,role) VALUES(?,?,?,?)",("admin",hashlib.sha256(b"admin123").hexdigest(),"Dr. Admin","admin"))
    cn.commit();cn.close()
def auth(u,p):
    cn=sqlite3.connect(DB);r=cn.cursor().execute("SELECT id,username,name,role FROM users WHERE username=? AND pw=?",(u,hashlib.sha256(p.encode()).hexdigest())).fetchone();cn.close();return r
def register(u,p,n):
    cn=sqlite3.connect(DB)
    try:cn.cursor().execute("INSERT INTO users(username,pw,name) VALUES(?,?,?)",(u,hashlib.sha256(p.encode()).hexdigest(),n));cn.commit();cn.close();return True
    except:cn.close();return False
def save_pred(uid,pj,pred,prob):
    cn=sqlite3.connect(DB);cn.cursor().execute("INSERT INTO predictions(user_id,patient_json,prediction,probability) VALUES(?,?,?,?)",(uid,pj,pred,prob));cn.commit();cn.close()
def get_history(uid):
    cn=sqlite3.connect(DB);df=pd.read_sql_query("SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 50",cn,params=(uid,));cn.close();return df
init_db()

# ═══ MODEL (UNTOUCHED) ═══════════════════════════════════════════════════
@st.cache_resource
def get_model_and_data():
    from scipy.io import arff
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    arff_path = None
    for p in [os.path.join(_HERE,"..","data","bone-marrow.arff"),os.path.join(_HERE,"bone-marrow.arff"),"data/bone-marrow.arff","bone-marrow.arff"]:
        if os.path.exists(p): arff_path = p; break
    if arff_path is None: return None, None, None, None, "ARFF file not found"
    data, _ = arff.loadarff(arff_path)
    df = pd.DataFrame(data)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    df = df.replace("?", np.nan)
    for col in df.columns:
        if df[col].isnull().sum() == 0: continue
        if df[col].dtype in ["float32","float64","int32","int64"]: df[col] = df[col].fillna(df[col].median())
        else: df[col] = df[col].fillna(df[col].mode()[0])
    numeric_cols = [c for c in df.select_dtypes(include=["float32","float64","int32","int64"]).columns if c != "survival_status"]
    clip_bounds = {}
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1; lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        clip_bounds[col] = (float(lo), float(hi))
        df[col] = df[col].clip(lower=lo, upper=hi)
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder(); le.fit(df[col].astype(str))
        df[col] = le.transform(df[col].astype(str))
    for col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    target = "survival_status"
    feature_cols = [c for c in df.columns if c != target]
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    df = df.drop(columns=to_drop)
    X = df.drop(columns=[target]); y = df[target].astype(int)
    features = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    try:
        from imblearn.over_sampling import SMOTE
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    except ImportError: pass
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)
    except ImportError:
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, max_depth=12)
        model.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    info = {"features": features, "clip_bounds": clip_bounds, "accuracy": acc, "auc": auc,
            "n_train": len(X_train), "n_test": len(X_test),
            "model_type": "XGBoost" if "xgb" in type(model).__name__.lower() else type(model).__name__.replace("Classifier",""),
            "X_full": X}
    return model, features, clip_bounds, info, None

@st.cache_data
def load_arff_raw():
    from scipy.io import arff
    for p in [os.path.join(_HERE,"..","data","bone-marrow.arff"),os.path.join(_HERE,"bone-marrow.arff"),"data/bone-marrow.arff","bone-marrow.arff"]:
        if os.path.exists(p):
            data,_=arff.loadarff(p);df=pd.DataFrame(data)
            for col in df.select_dtypes(include=["object"]).columns:
                df[col]=df[col].apply(lambda x:x.decode("utf-8") if isinstance(x,bytes) else x)
            return df
    return None

# ═══ FOOTER HELPER ════════════════════════════════════════════════════════
def render_footer():
    st.markdown("""
    <div class="bmt-ft">
      <div class="ft-in">
        <div class="ft-logo">BMT.AI &nbsp;&mdash;&nbsp; Bone Marrow Transplant Predictor</div>
        <div class="ft-div"></div>
        <div class="ft-main">
          <strong>Centrale Casablanca</strong> &nbsp;&middot;&nbsp;
          Coding Week &nbsp;&middot;&nbsp; March 2026 &nbsp;&middot;&nbsp;
          <strong>Team 19</strong> &nbsp;&middot;&nbsp; k.&nbsp;Zerhouni; R.&nbsp;Nassih
        </div>
        <div class="ft-credit">
          Designed &amp; developed by <strong>Abderrahman Gouiferda</strong>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ═══ SESSION STATE ════════════════════════════════════════════════════════
for k,v in [("logged_in",False),("user",None),("page","predict"),("welcomed",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ═══ WELCOME PAGE ═════════════════════════════════════════════════════════
def page_welcome():
    st.markdown("""
    <div class="welcome-wrap">
        <div style="font-size:64px;margin-bottom:16px">🧬</div>
        <div class="welcome-title">Welcome to <span>BMT.AI</span></div>
        <div class="welcome-sub">
            Revolutionizing pediatric bone marrow transplant outcomes with intelligent analytics and explainable machine learning
        </div>
        <div class="welcome-features">
            <div class="wf-card"><div class="wf-icon">🔮</div><div class="wf-label">Prediction</div></div>
            <div class="wf-card"><div class="wf-icon">🧠</div><div class="wf-label">Explainability</div></div>
            <div class="wf-card"><div class="wf-icon">📊</div><div class="wf-label">Analytics</div></div>
            <div class="wf-card"><div class="wf-icon">🔒</div><div class="wf-label">Secure</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    _,bc,_ = st.columns([1,1,1])
    with bc:
        if st.button("Explore Dashboard →", use_container_width=True, key="welcome_btn"):
            st.session_state.welcomed = True
            st.rerun()

    render_footer()

# ═══ LOGIN PAGE ═══════════════════════════════════════════════════════════
def page_login():
    st.markdown("""<div style="text-align:center;padding:50px 0 0">
        <div style="font-size:52px;margin-bottom:8px">🧬</div>
        <h1 style="font-size:26px;color:#f0f4ff!important;margin-bottom:2px!important">BMT.AI</h1>
        <p style="color:#94a3b8;font-size:14px;margin-bottom:40px">Secure Login — Medical Decision Support</p>
    </div>""", unsafe_allow_html=True)
    _,col,_=st.columns([1,1.3,1])
    with col:
        st.markdown('<div class="glass" style="padding:36px 32px">', unsafe_allow_html=True)
        t1,t2=st.tabs(["🔑 Sign In","📝 Register"])
        with t1:
            u=st.text_input("Username",key="lu",placeholder="Enter username")
            p=st.text_input("Password",type="password",key="lp",placeholder="Enter password")
            if st.button("Secure Login",key="lb",use_container_width=True):
                r=auth(u,p)
                if r:st.session_state.logged_in=True;st.session_state.user={"id":r[0],"username":r[1],"name":r[2],"role":r[3]};st.rerun()
                else:st.error("Invalid credentials.")
            st.markdown("<p style='text-align:center;color:#64748b;font-size:12px;margin-top:12px'>Default: <b style=\"color:#94a3b8\">admin</b> / <b style=\"color:#94a3b8\">admin123</b></p>",unsafe_allow_html=True)
        with t2:
            rn=st.text_input("Full Name",key="rn");ru=st.text_input("Username",key="ru2")
            rp=st.text_input("Password",type="password",key="rp");rp2=st.text_input("Confirm",type="password",key="rp2")
            if st.button("Create Account",key="rb",use_container_width=True):
                if rp!=rp2:st.error("Passwords don't match.")
                elif register(ru,rp,rn):st.success("Account created!")
                else:st.error("Username taken.")
        st.markdown('</div>', unsafe_allow_html=True)
    render_footer()

# ═══ SIDEBAR ══════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        u=st.session_state.user
        st.markdown(f"""<div style="padding:22px 0;text-align:center;border-bottom:1px solid rgba(59,130,246,0.12);margin-bottom:18px">
            <div style="font-size:36px;margin-bottom:4px">🧬</div>
            <div style="font-size:17px;font-weight:800;color:#60a5fa">BMT.AI</div>
            <div style="font-size:10.5px;opacity:0.5;color:#94a3b8">Decision Support</div>
        </div>
        <div style="padding:9px 13px;background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.15);border-radius:10px;margin-bottom:18px">
            <div style="font-size:12.5px;font-weight:600;color:#c7d2fe">👋 {u['name']}</div>
            <div style="font-size:10.5px;color:#64748b">{u['role'].title()} · @{u['username']}</div>
        </div>""",unsafe_allow_html=True)
        st.markdown("---")
        for key,label in {"predict":"🔮 New Prediction","history":"📋 History","dashboard":"📊 Dashboard","shap":"🧠 Explainability","data":"📁 Dataset"}.items():
            if st.button(label,key=f"n_{key}",use_container_width=True):st.session_state.page=key;st.rerun()
        st.markdown("---")
        if st.button("🚪 Sign Out",key="logout",use_container_width=True):st.session_state.logged_in=False;st.session_state.user=None;st.rerun()


# ═══ TOGGLE HELPER — iOS-style Yes/No toggle ═════════════════════════════
def toggle(label, key, default=False):
    """Render an iOS-style toggle switch using st.checkbox with custom CSS."""
    val = st.checkbox(label, value=default, key=key)
    return 1 if val else 0

# ═══ PREDICTION PAGE ═════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div class="hero"><div class="gp">Prediction Engine</div><h1>🔮 Transplant Outcome Prediction</h1><p>Enter patient &amp; clinical data to predict transplant success</p></div>',unsafe_allow_html=True)

    # Inject toggle switch CSS for checkboxes
    st.markdown("""<style>
    /* iOS Toggle Switch for checkboxes */
    .stCheckbox > label > div[data-testid="stCheckbox"] > div:first-child {
        display: none !important;
    }
    .stCheckbox label {
        position: relative; cursor: pointer; display: flex; align-items: center; gap: 12px;
    }
    .stCheckbox > label > div[role="checkbox"] {
        width: 48px !important; height: 26px !important;
        background: #334155 !important;
        border-radius: 26px !important;
        position: relative !important;
        transition: background 0.3s !important;
        border: none !important;
        min-height: 26px !important;
    }
    .stCheckbox > label > div[role="checkbox"][aria-checked="true"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    }
    .stCheckbox > label > div[role="checkbox"]::after {
        content: '';
        position: absolute;
        width: 20px; height: 20px;
        background: white;
        border-radius: 50%;
        top: 3px; left: 3px;
        transition: transform 0.3s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stCheckbox > label > div[role="checkbox"][aria-checked="true"]::after {
        transform: translateX(22px);
    }
    .stCheckbox > label > span {
        color: #c7d2fe !important; font-size: 13px !important; font-weight: 500 !important;
    }

    /* Slider track style - modern line with circle */
    [data-baseweb="slider"] {
        padding-top: 14px !important;
    }
    [data-baseweb="slider"] [role="slider"] {
        width: 22px !important; height: 22px !important;
        background: white !important;
        border: 3px solid #3b82f6 !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 10px rgba(59,130,246,0.4) !important;
    }
    [data-baseweb="slider"] > div > div:nth-child(3) {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        height: 4px !important;
    }
    [data-baseweb="slider"] > div > div:nth-child(4) {
        background: #334155 !important;
        height: 4px !important;
    }
    .stSlider label { color: #94a3b8 !important; font-size: 13px !important; font-weight: 600 !important; }
    .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { color: #64748b !important; }
    </style>""", unsafe_allow_html=True)

    model, features, clip_bounds, info, err = get_model_and_data()
    if err:
        st.error(f"⚠️ {err}. Place `bone-marrow.arff` in `data/` folder or next to this file.")
        return
    st.markdown(f'<div class="card">✅ Model ready: <b style="color:#60a5fa">{info["model_type"]}</b> trained on {info["n_train"]} samples | Accuracy: {info["accuracy"]:.1%} | ROC-AUC: {info["auc"]:.3f}</div>',unsafe_allow_html=True)

    inp={}

    with st.expander("👤 Patient Demographics",expanded=True):
        c1,c2,c3=st.columns(3)
        inp["Recipientgender"]=1 if c1.selectbox("Recipient Gender",["Female","Male"],index=1,key="f1")=="Male" else 0
        inp["Recipientage"]=c2.slider("Recipient Age (yr)",0.5,21.0,9.6,0.1,key="f2")
        inp["Rbodymass"]=c3.slider("Body Mass (kg)",3.0,100.0,33.0,0.5,key="f3")

    with st.expander("🧬 Donor Information",expanded=True):
        c1,c2,c3=st.columns(3)
        inp["Stemcellsource"]=1 if c1.selectbox("Stem Cell Source",["Bone Marrow","Peripheral Blood"],index=1,key="f4")=="Peripheral Blood" else 0
        inp["Donorage"]=c2.slider("Donor Age (yr)",18.0,60.0,33.5,0.1,key="f5")
        with c3: inp["Gendermatch"]=toggle("Female→Male Gender Match",key="f6")

    with st.expander("🩸 Blood & Immunology",expanded=True):
        c1,c2,c3=st.columns(3)
        abo={"B":0,"O":1,"A":2,"AB":3}
        inp["DonorABO"]=abo[c1.selectbox("Donor ABO",["O","A","B","AB"],key="f7")]
        inp["RecipientABO"]=abo[c2.selectbox("Recipient ABO",["O","A","B","AB"],key="f8")]
        inp["RecipientRh"]=1 if c3.selectbox("Recipient Rh",["Rh−","Rh+"],index=1,key="f9")=="Rh+" else 0
        c1,c2=st.columns(2)
        with c1: inp["ABOmatch"]=toggle("ABO Match",key="f10",default=True)
        inp["CMVstatus"]={"Fully Compatible":0,"Level 1":1,"Level 2":2,"Least Compatible":3}[c2.selectbox("CMV Compatibility",["Fully Compatible","Level 1","Level 2","Least Compatible"],index=2,key="f11")]
        c1,c2=st.columns(2)
        with c1: inp["DonorCMV"]=toggle("Donor CMV Present",key="f12")
        with c2: inp["RecipientCMV"]=toggle("Recipient CMV Present",key="f13",default=True)

    with st.expander("🏥 Disease & Risk",expanded=True):
        c1,c2,c3=st.columns(3)
        inp["Disease"]={"ALL":0,"AML":1,"Chronic":2,"Lymphoma":3,"Non-malignant":4}[c1.selectbox("Disease",["ALL","AML","Chronic","Lymphoma","Non-malignant"],index=1,key="f14")]
        with c2: inp["Riskgroup"]=toggle("High Risk",key="f15")
        inp["Diseasegroup"]=1 if c3.selectbox("Disease Group",["Non-malignant","Malignant"],index=1,key="f16")=="Malignant" else 0
        c1,c2=st.columns(2)
        with c1: inp["Txpostrelapse"]=toggle("Post-Relapse Transplant",key="f17")
        with c2: inp["Relapse"]=toggle("Relapse",key="f18")

    with st.expander("🔗 HLA Compatibility",expanded=True):
        c1,c2,c3,c4=st.columns(4)
        inp["HLAmatch"]={"10/10":0,"9/10":1,"8/10":2,"7/10":3}[c1.selectbox("HLA Match",["10/10","9/10","8/10","7/10"],key="f19")]
        inp["HLAmismatch"]=0 if inp["HLAmatch"]==0 else 1
        inp["Antigen"]={"None":0,"1":1,"2":2,"3":3}[c2.selectbox("Antigen Diffs",["None","1","2","3"],key="f20")]
        inp["Alel"]={"None":0,"1":1,"2":2,"3":3,"4":4}[c3.selectbox("Allele Diffs",["None","1","2","3","4"],key="f21")]
        inp["HLAgrI"]={"Matched":0,"1 Antigen":1,"1 Allele":2,"DRB1":3,"2 Diffs":4,"2+":5}[c4.selectbox("HLA Gr.I",["Matched","1 Antigen","1 Allele","DRB1","2 Diffs","2+"],key="f22")]

    with st.expander("💉 Cell Doses",expanded=True):
        c1,c2,c3=st.columns(3)
        inp["CD34kgx10d6"]=c1.slider("CD34+ (×10⁶/kg)",0.5,60.0,9.7,0.1,key="f23")
        inp["CD3dCD34"]=c2.slider("CD3/CD34 Ratio",0.1,100.0,2.7,0.1,key="f24")
        inp["CD3dkgx10d8"]=c3.slider("CD3+ (×10⁸/kg)",0.01,20.0,4.3,0.01,key="f25")

    with st.expander("⏱ Recovery & GvHD",expanded=True):
        c1,c2,c3=st.columns(3)
        with c1: inp["IIIV"]=toggle("Acute GvHD II-IV",key="f26",default=True)
        with c2: inp["aGvHDIIIIV"]=1-toggle("Acute GvHD III/IV",key="f27")
        with c3: inp["extcGvHD"]=1-toggle("Ext. Chronic GvHD",key="f28")
        c1,c2=st.columns(2)
        inp["ANCrecovery"]=c1.slider("ANC Recovery (days)",7.0,100.0,15.0,1.0,key="f29",help="Neutrophil recovery. Clipped to 7-23 by model.")
        inp["PLTrecovery"]=c2.slider("PLT Recovery (days)",1.0,200.0,21.0,1.0,key="f30",help="Platelet recovery. Clipped to 9-69 by model.")
        c1,c2=st.columns(2)
        inp["time_to_aGvHD_III_IV"]=c1.number_input("Time to aGvHD III/IV",1.0,1000000.0,1000000.0,1.0,key="f31",help="1000000 if not developed")
        inp["survival_time"]=c2.slider("Survival Time (days)",1.0,4000.0,676.0,1.0,key="f32")

    # Auto-derived (UNTOUCHED)
    inp["Donorage35"]=1 if inp["Donorage"]>=35 else 0
    inp["Recipientage10"]=1 if inp["Recipientage"]>=10 else 0
    a=inp["Recipientage"];inp["Recipientageint"]=0 if a<=5 else (1 if a<=10 else 2)

    with st.expander("🔧 Auto-Computed",expanded=False):
        ac=st.columns(4)
        ac[0].metric("Donor≥35","Yes" if inp["Donorage35"] else "No")
        ac[1].metric("Recip≥10","Yes" if inp["Recipientage10"] else "No")
        ac[2].metric("Age Grp",{0:"0-5",1:"5-10",2:"10-20"}[inp["Recipientageint"]])
        ac[3].metric("HLA Mis","Yes" if inp["HLAmismatch"] else "No")

    st.markdown("---")
    _,bc,_=st.columns([1,2,1])
    with bc:clicked=st.button("🔮  Generate Prediction",use_container_width=True,key="pbtn")

    if clicked:
        # BACKEND UNTOUCHED
        fd={f:inp.get(f,0) for f in features}
        fv=pd.DataFrame([fd])
        for col in fv.columns:fv[col]=pd.to_numeric(fv[col],errors="coerce").fillna(0)
        for col,(lo,hi) in clip_bounds.items():
            if col in fv.columns:fv[col]=fv[col].clip(lower=lo,upper=hi)
        pred=int(model.predict(fv)[0])
        proba=model.predict_proba(fv)[0]
        sp=float(proba[0])*100;fp=float(proba[1])*100
        pj=json.dumps({k:round(float(v),4) if isinstance(v,(float,np.floating)) else int(v) for k,v in fd.items()})
        save_pred(st.session_state.user["id"],pj,pred,max(sp,fp))

        st.markdown("---")
        if pred==0:
            st.markdown(f'<div class="pg"><div style="font-size:42px;margin-bottom:8px">✅</div><h2>Favorable — Likely Survived</h2><p style="color:#86efac;font-size:14px">Predicted <b>positive outcome</b> with {sp:.1f}% confidence</p><div class="bar"><div class="bf" style="width:{sp}%;background:linear-gradient(90deg,#22c55e,#16a34a)"></div></div></div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="pb"><div style="font-size:42px;margin-bottom:8px">⚠️</div><h2>High Risk — Not Survived</h2><p style="color:#fca5a5;font-size:14px">Predicted <b>negative outcome</b> with {fp:.1f}% confidence</p><div class="bar"><div class="bf" style="width:{fp}%;background:linear-gradient(90deg,#ef4444,#dc2626)"></div></div></div>',unsafe_allow_html=True)

        p1,p2=st.columns(2)
        p1.markdown(f'<div class="mc"><div class="ml">Survival</div><div class="mv" style="color:#22c55e">{sp:.1f}%</div></div>',unsafe_allow_html=True)
        p2.markdown(f'<div class="mc"><div class="ml">Non-Survival</div><div class="mv" style="color:#ef4444">{fp:.1f}%</div></div>',unsafe_allow_html=True)

        # SHAP-like explanation (UNTOUCHED logic, updated colors)
        st.markdown("#### 🧠 Feature Impact Explanation")
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        X_float = info["X_full"].astype(float)
        base_p = proba[1]
        contributions = {}
        for i, feat in enumerate(features):
            mod = fv.copy().astype(float)
            mod.iloc[0, i] = float(X_float.iloc[:, i].mean())
            new_p = model.predict_proba(mod)[0][1]
            contributions[feat] = base_p - new_p
        sorted_c = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:12]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0a1628'); ax.set_facecolor('#0a1628')
        cnames = [c[0] for c in sorted_c]; cvals = [c[1] for c in sorted_c]
        ccolors = ['#ef4444' if v > 0 else '#22c55e' for v in cvals]
        bars = ax.barh(range(len(cnames)), cvals, color=ccolors, height=0.6, edgecolor='#1a3a5c', linewidth=0.5)
        for j, (b, v) in enumerate(zip(bars, cvals)):
            ha = 'left' if v > 0 else 'right'
            ax.text(v + (0.003 if v > 0 else -0.003), j, f"{v:+.3f}", ha=ha, va='center', fontsize=9, color='#94a3b8', fontweight='600')
        ax.set_yticks(range(len(cnames))); ax.set_yticklabels(cnames, fontsize=10, fontweight='500', color='#c7d2fe')
        ax.invert_yaxis(); ax.axvline(0, color='#334155', linewidth=1)
        ax.set_xlabel("← Survival                    Impact                    Non-Survival →", fontsize=10, fontweight='600', color='#64748b')
        ax.set_title(f"What Drives This Prediction — P(survived) = {sp:.1f}%", fontsize=14, fontweight='800', color='#f0f4ff', pad=15)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color('#1e3a5f'); ax.spines["bottom"].set_color('#1e3a5f')
        ax.tick_params(colors='#64748b')
        ax.legend(handles=[mpatches.Patch(color='#22c55e',label='→ Survival'), mpatches.Patch(color='#ef4444',label='→ Non-Survival')], loc='lower right', fontsize=9, framealpha=0.3, facecolor='#0a1628', edgecolor='#1e3a5f', labelcolor='#94a3b8')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="wb"><strong>⚠️ Disclaimer:</strong> Decision support only. Clinical decisions must be made by qualified professionals.</div>',unsafe_allow_html=True)

    render_footer()


# ═══ HISTORY (logic untouched, colors updated) ════════════════════════════
def page_history():
    st.markdown('<div class="hero"><div class="gp">Records</div><h1>📋 Prediction History</h1><p>Review past predictions</p></div>',unsafe_allow_html=True)
    h=get_history(st.session_state.user["id"])
    if h.empty:st.info("No predictions yet.");render_footer();return
    t=len(h);s=(h["prediction"]==0).sum();n=(h["prediction"]==1).sum();a=h["probability"].mean()
    st.markdown(f'<div class="mrow"><div class="mc"><div class="ml">Total</div><div class="mv" style="color:#60a5fa">{t}</div></div><div class="mc"><div class="ml">Survived</div><div class="mv" style="color:#22c55e">{s}</div></div><div class="mc"><div class="ml">Not Survived</div><div class="mv" style="color:#ef4444">{n}</div></div><div class="mc"><div class="ml">Avg Conf</div><div class="mv">{a:.1f}%</div></div></div>',unsafe_allow_html=True)
    d=h[["id","prediction","probability","created_at"]].copy();d["prediction"]=d["prediction"].map({0:"✅ Survived",1:"⚠️ Not Survived"});d["probability"]=d["probability"].round(1).astype(str)+"%";d.columns=["#","Prediction","Confidence","Date"]
    st.dataframe(d,use_container_width=True,hide_index=True)
    render_footer()

# ═══ DASHBOARD (logic untouched, colors updated) ═════════════════════════
def page_dashboard():
    st.markdown('<div class="hero"><div class="gp">Analytics</div><h1>📊 Model Performance</h1><p>Metrics and comparison</p></div>',unsafe_allow_html=True)
    _,_,_,info,err=get_model_and_data()
    if err:st.error(err);return
    st.markdown("### Model Comparison")
    dc=pd.DataFrame({"Model":["Random Forest","XGBoost","LightGBM"],"Accuracy":[0.8537,0.8780,0.8780],"Precision":[0.7857,0.8182,0.8571],"Recall":[0.8462,0.6923,0.6923],"F1":[0.8148,0.7500,0.7660],"ROC-AUC":[0.9122,0.9136,0.9073]})
    st.dataframe(dc.style.highlight_max(subset=["Accuracy","Precision","Recall","F1","ROC-AUC"],color="rgba(59,130,246,0.3)"),use_container_width=True,hide_index=True)
    st.markdown(f'<div class="card">✅ <b style="color:#60a5fa">Current: {info["model_type"]}</b> — Accuracy: {info["accuracy"]:.1%}, ROC-AUC: {info["auc"]:.3f}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="mrow"><div class="mc"><div class="ml">Model</div><div class="mv" style="color:#60a5fa;font-size:16px">{info["model_type"]}</div></div><div class="mc"><div class="ml">Features</div><div class="mv">{len(info["features"])}</div></div><div class="mc"><div class="ml">Accuracy</div><div class="mv">{info["accuracy"]:.1%}</div></div><div class="mc"><div class="ml">ROC-AUC</div><div class="mv">{info["auc"]:.3f}</div></div></div>',unsafe_allow_html=True)
    try:
        import matplotlib.pyplot as plt;import matplotlib;matplotlib.use("Agg")
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(6,3.5));fig.patch.set_facecolor('#0a1628');ax.set_facecolor('#0a1628')
            bars=ax.barh(["RF","XGBoost","LightGBM"],[0.9122,0.9136,0.9073],color=["#3b82f6","#8b5cf6","#ec4899"],height=0.45);ax.set_xlim(0.89,0.92)
            for b,v in zip(bars,[0.9122,0.9136,0.9073]):ax.text(v+0.0003,b.get_y()+b.get_height()/2,f"{v:.4f}",va="center",fontweight="bold",fontsize=10,color="#c7d2fe")
            ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False);ax.spines["left"].set_color('#1e3a5f');ax.spines["bottom"].set_color('#1e3a5f')
            ax.tick_params(colors='#94a3b8');ax.set_xlabel("ROC-AUC",color="#94a3b8");plt.tight_layout();st.pyplot(fig);plt.close()
        with c2:
            fig2,ax2=plt.subplots(figsize=(6,3.5));fig2.patch.set_facecolor('#0a1628');ax2.set_facecolor('#0a1628')
            x=np.arange(4);w=0.3
            ax2.bar(x-w/2,[0.854,0.786,0.846,0.815],w,label="RF",color="#3b82f6");ax2.bar(x+w/2,[0.878,0.818,0.692,0.750],w,label="XGB",color="#8b5cf6")
            ax2.set_xticks(x);ax2.set_xticklabels(["Acc","Prec","Rec","F1"],color="#94a3b8");ax2.set_ylim(0.5,1.0);ax2.legend(fontsize=9,facecolor='#0a1628',edgecolor='#1e3a5f',labelcolor='#94a3b8')
            ax2.spines["top"].set_visible(False);ax2.spines["right"].set_visible(False);ax2.spines["left"].set_color('#1e3a5f');ax2.spines["bottom"].set_color('#1e3a5f')
            ax2.tick_params(colors='#94a3b8');plt.tight_layout();st.pyplot(fig2);plt.close()
    except:pass
    st.markdown("### Pipeline\n1. **Missing** → median/mode\n2. **Outliers** → IQR clip\n3. **Encoding** → LabelEncoder\n4. **Correlation** → Drop >0.95\n5. **SMOTE** → Balance\n6. **Train** → XGBoost (or RandomForest)")
    render_footer()

# ═══ SHAP PAGE (logic untouched, colors updated) ═════════════════════════
def page_shap():
    st.markdown('<div class="hero"><div class="gp">Explainability</div><h1>🧠 SHAP Explainability</h1><p>Understand which features drive predictions</p></div>',unsafe_allow_html=True)
    model,features,_,info,err=get_model_and_data()
    if err:st.error(err);return
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    X = info["X_full"]
    try:
        df_raw = load_arff_raw()
        y_vals = pd.to_numeric(df_raw["survival_status"], errors="coerce").fillna(0).astype(int).values[:len(X)] if df_raw is not None else np.zeros(len(X))
    except: y_vals = np.zeros(len(X))
    imp = model.feature_importances_

    st.markdown("### Global Feature Importance (Gini)")
    sorted_idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0a1628'); ax.set_facecolor('#0a1628')
    names = [features[i] for i in sorted_idx]; vals = [imp[i] for i in sorted_idx]; max_v = max(vals)
    colors = [f'#{int(59+v/max_v*100):02x}{int(130+v/max_v*60):02x}{int(246):02x}' for v in vals]
    bars = ax.barh(range(len(names)), vals, color=colors, height=0.65, edgecolor='#1a3a5c', linewidth=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=11, fontweight='500', color='#c7d2fe')
    ax.invert_yaxis(); ax.set_xlabel("Feature Importance", fontsize=12, fontweight='600', color='#64748b')
    ax.set_title("Global Feature Importance", fontsize=16, fontweight='800', color='#f0f4ff', pad=18)
    for b, v in zip(bars, vals):
        ax.text(v + max_v*0.01, b.get_y()+b.get_height()/2, f"{v:.4f}", va="center", fontsize=9, color="#94a3b8", fontweight='500')
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color('#1e3a5f'); ax.spines["bottom"].set_color('#1e3a5f'); ax.tick_params(colors='#64748b')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("### Feature Impact Direction")
    st.markdown("Each dot = one patient. **Red** = high value, **Blue** = low. Position = impact direction.")
    top_n = 12; top_idx = sorted_idx[:top_n]
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a1628'); ax.set_facecolor('#0a1628')
    for row, idx in enumerate(top_idx):
        fv = X.iloc[:, idx].values.astype(float); fmin, fmax = fv.min(), fv.max()
        nv = (fv - fmin) / (fmax - fmin) if fmax > fmin else np.zeros_like(fv)
        corr = np.corrcoef(fv, y_vals)[0, 1] if np.std(fv) > 0 and np.std(y_vals) > 0 else 0
        if np.isnan(corr): corr = 0
        jitter = np.random.normal(0, 0.12, len(fv))
        x_pos = corr * imp[idx] * nv + np.random.normal(0, imp[idx]*0.05, len(fv))
        ax.scatter(x_pos, np.full(len(fv), row) + jitter, c=nv, cmap='RdBu_r', s=6, alpha=0.7, vmin=0, vmax=1, edgecolors='none')
    ax.set_yticks(range(top_n)); ax.set_yticklabels([features[i] for i in top_idx], fontsize=11, fontweight='500', color='#c7d2fe')
    ax.invert_yaxis(); ax.axvline(0, color='#334155', linewidth=0.8)
    ax.set_xlabel("← Survival          Impact          Non-Survival →", fontsize=11, fontweight='600', color='#64748b')
    ax.set_title("Feature Impact Direction (SHAP-like)", fontsize=16, fontweight='800', color='#f0f4ff', pad=18)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color('#1e3a5f'); ax.spines["bottom"].set_color('#1e3a5f'); ax.tick_params(colors='#64748b')
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0,1)); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Feature Value', fontsize=10, color='#94a3b8'); cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(['Low', 'Mid', 'High'])
    cbar.ax.tick_params(colors='#94a3b8')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("### Top 5 Most Influential Features")
    lm = {"survival_time":"Survival Time","ANCrecovery":"ANC Recovery","PLTrecovery":"Platelet Recovery","Relapse":"Disease Relapse","CD34kgx10d6":"CD34+ Cell Dose","Recipientage":"Recipient Age","Donorage":"Donor Age","Rbodymass":"Body Mass","Riskgroup":"Risk Group","CD3dkgx10d8":"CD3+ Dose","CD3dCD34":"CD3/CD34 Ratio","Disease":"Disease Type"}
    for idx in sorted_idx[:5]:
        f = features[idx]; label = lm.get(f, f)
        st.markdown(f'<div class="card"><b style="color:#60a5fa">{label}</b> ({f}) — <span class="pill pill-g">Importance = {imp[idx]:.4f}</span></div>', unsafe_allow_html=True)
    render_footer()

# ═══ DATASET (logic untouched, colors updated) ═══════════════════════════
def page_data():
    st.markdown('<div class="hero"><div class="gp">Explorer</div><h1>📁 Dataset Explorer</h1><p>Explore the training data</p></div>',unsafe_allow_html=True)
    df=load_arff_raw()
    if df is None:st.error("ARFF not found.");return
    tg="survival_status";df[tg]=pd.to_numeric(df[tg],errors="coerce").fillna(0).astype(int)
    ns=int((df[tg]==0).sum());nn=int((df[tg]==1).sum())
    st.markdown(f'<div class="mrow"><div class="mc"><div class="ml">Samples</div><div class="mv" style="color:#60a5fa">{len(df)}</div></div><div class="mc"><div class="ml">Features</div><div class="mv">{len(df.columns)-1}</div></div><div class="mc"><div class="ml">Survived</div><div class="mv" style="color:#22c55e">{ns}</div></div><div class="mc"><div class="ml">Not Survived</div><div class="mv" style="color:#ef4444">{nn}</div></div></div>',unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["📊 Preview","📈 Stats","🔍 Analysis"])
    with t1:st.dataframe(df.head(20),use_container_width=True,hide_index=True)
    with t2:
        num=df.select_dtypes(include=[np.number])
        if not num.empty:st.dataframe(num.describe().round(3),use_container_width=True)
    with t3:
        cols=[c for c in df.columns if c!=tg]
        sel=st.selectbox("Feature",cols)
        if sel:
            try:
                import matplotlib.pyplot as plt;import matplotlib;matplotlib.use("Agg")
                pd_=pd.to_numeric(df[sel],errors="coerce").dropna()
                c1,c2=st.columns(2)
                with c1:
                    fig,ax=plt.subplots(figsize=(6,3.5));fig.patch.set_facecolor('#0a1628');ax.set_facecolor('#0a1628')
                    pd_.hist(bins=30,ax=ax,color="#3b82f6",edgecolor="#1a3a5c",alpha=0.85);ax.set_title(sel,fontweight="bold",color="#f0f4ff")
                    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False);ax.spines["left"].set_color('#1e3a5f');ax.spines["bottom"].set_color('#1e3a5f');ax.tick_params(colors='#94a3b8')
                    plt.tight_layout();st.pyplot(fig);plt.close()
                with c2:
                    fig2,ax2=plt.subplots(figsize=(6,3.5));fig2.patch.set_facecolor('#0a1628');ax2.set_facecolor('#0a1628')
                    tgt=df[tg]
                    ax2.hist(pd_[tgt==0],bins=25,alpha=0.7,label="Surv",color="#22c55e");ax2.hist(pd_[tgt==1],bins=25,alpha=0.7,label="Not",color="#ef4444")
                    ax2.legend(facecolor='#0a1628',edgecolor='#1e3a5f',labelcolor='#94a3b8');ax2.set_title(f"By Outcome: {sel}",fontweight="bold",color="#f0f4ff")
                    ax2.spines["top"].set_visible(False);ax2.spines["right"].set_visible(False);ax2.spines["left"].set_color('#1e3a5f');ax2.spines["bottom"].set_color('#1e3a5f');ax2.tick_params(colors='#94a3b8')
                    plt.tight_layout();st.pyplot(fig2);plt.close()
            except Exception as e:st.warning(f"Chart: {e}")
    render_footer()

# ═══ MAIN ROUTER ══════════════════════════════════════════════════════════
def main():
    if not st.session_state.welcomed:
        page_welcome()
    elif not st.session_state.logged_in:
        page_login()
    else:
        sidebar()
        {"predict":page_predict,"history":page_history,"dashboard":page_dashboard,"shap":page_shap,"data":page_data}.get(st.session_state.page,page_predict)()

if __name__=="__main__":main()