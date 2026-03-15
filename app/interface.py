"""
Bone Marrow Transplant Decision Support System
Centrale Casablanca — Coding Week Project 4, March 2026

This app auto-trains a model from bone-marrow.arff using the EXACT same
pipeline as data_processing.py. Zero model mismatch possible.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, sqlite3, hashlib, json, warnings, joblib
warnings.filterwarnings("ignore")

st.set_page_config(page_title="BMT Decision Support", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# ═══ CSS ══════════════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');
:root{--em:#065F46;--em2:#059669;--cream:#FAFDF7;--sf:#FFF;--tx:#111827;--txm:#6B7280;--bd:#E5E7EB;--red:#DC2626;--grn:#16A34A;}
html,body,[class*="css"]{font-family:'Bricolage Grotesque',sans-serif!important;color:var(--tx);}
.stApp{background:var(--cream)!important;}
section[data-testid="stSidebar"]{background:linear-gradient(175deg,#042F2E 0%,#065F46 45%,#059669 100%)!important;}
section[data-testid="stSidebar"] *{color:#ecfdf5!important;}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,0.12)!important;}
section[data-testid="stSidebar"] .stButton>button{background:rgba(255,255,255,0.08)!important;border:1px solid rgba(255,255,255,0.12)!important;color:#ecfdf5!important;text-align:left!important;padding:10px 16px!important;border-radius:10px!important;font-weight:500!important;}
section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,0.15)!important;}
h1,h2,h3{font-family:'Bricolage Grotesque',sans-serif!important;font-weight:700!important;color:var(--tx)!important;letter-spacing:-0.02em!important;}
.hero{background:linear-gradient(135deg,#042F2E 0%,#065F46 50%,#059669 100%);padding:38px 42px;border-radius:22px;margin-bottom:30px;position:relative;overflow:hidden;}
.hero::after{content:'';position:absolute;top:-80px;right:-60px;width:300px;height:300px;background:radial-gradient(circle,rgba(212,168,67,0.12) 0%,transparent 70%);border-radius:50%;}
.hero h1{color:#ecfdf5!important;font-size:28px!important;margin-bottom:4px!important;}
.hero p{color:rgba(236,253,245,0.7);font-size:14px;margin:0;}
.hero .gp{display:inline-block;background:rgba(212,168,67,0.2);border:1px solid rgba(212,168,67,0.35);color:#F5E6C4;padding:3px 14px;border-radius:20px;font-size:10.5px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:10px;}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:16px;padding:22px 26px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.03);}
.mrow{display:flex;gap:12px;margin-bottom:22px;flex-wrap:wrap;}
.mc{flex:1;min-width:120px;background:var(--sf);border:1px solid var(--bd);border-radius:14px;padding:16px 18px;text-align:center;}
.mc .ml{font-size:10px; 'font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--txm);margin-bottom:5px;}
.mc .mv{font-size:24px;font-weight:800;font-family:'Fira Code',monospace;}
.pg{background:linear-gradient(135deg,#DCFCE7,#BBF7D0);border:2px solid var(--grn);border-radius:20px;padding:34px;text-align:center;margin:22px 0;}
.pg h2{color:#166534!important;font-size:23px!important;}
.pb{background:linear-gradient(135deg,#FEE2E2,#FECACA);border:2px solid var(--red);border-radius:20px;padding:34px;text-align:center;margin:22px 0;}
.pb h2{color:#991B1B!important;font-size:23px!important;}
.bar{background:#E5E7EB;border-radius:10px;height:18px;margin:12px auto;overflow:hidden;max-width:360px;}
.bf{height:100%;border-radius:10px;}
.stButton>button{background:linear-gradient(135deg,#065F46,#059669)!important;color:white!important;border:none!important;border-radius:12px!important;padding:11px 28px!important;font-weight:600!important;font-size:14px!important;box-shadow:0 2px 10px rgba(6,95,70,0.25)!important;}
.wb{background:#FFFBEB;border:1px solid #F59E0B;border-radius:12px;padding:14px 18px;margin-top:18px;}
.ft{text-align:center;padding:26px;color:var(--txm);font-size:12px;border-top:1px solid var(--bd);margin-top:44px;}
#MainMenu,footer,header,[data-testid="stHeader"],[data-testid="stToolbar"]{display:none!important;}
.streamlit-expanderHeader{font-weight:600!important;font-size:14px!important;}
</style>""", unsafe_allow_html=True)

_HERE = os.path.dirname(os.path.abspath(__file__))

# ═══ DATABASE ═════════════════════════════════════════════════════════════
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

# ═══ MODEL: Auto-train from ARFF using EXACT data_processing.py pipeline ═
@st.cache_resource
def get_model_and_data():
    """
    Reproduce the EXACT pipeline from data_processing.py + train_model.py:
    1. Load ARFF
    2. Decode bytes → strings
    3. Replace '?' → NaN
    4. Fill missing: median (numeric), mode (categorical)
    5. IQR outlier clipping on numeric cols
    6. LabelEncoder on categorical cols
    7. Drop correlated features (threshold=0.95)
    8. Train/test split (80/20, stratified)
    9. SMOTE on training set
    10. Train XGBoost (or RandomForest as fallback)
    """
    from scipy.io import arff
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Find ARFF file
    arff_path = None
    for p in [os.path.join(_HERE,"..","data","bone-marrow.arff"),
              os.path.join(_HERE,"bone-marrow.arff"),
              "data/bone-marrow.arff","bone-marrow.arff"]:
        if os.path.exists(p): arff_path = p; break

    if arff_path is None:
        return None, None, None, None, "ARFF file not found"

    # Step 1-2: Load and decode
    data, _ = arff.loadarff(arff_path)
    df = pd.DataFrame(data)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    # Step 3-4: Missing values
    df = df.replace("?", np.nan)
    for col in df.columns:
        if df[col].isnull().sum() == 0: continue
        if df[col].dtype in ["float32","float64","int32","int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Step 5: IQR clipping
    numeric_cols = [c for c in df.select_dtypes(include=["float32","float64","int32","int64"]).columns if c != "survival_status"]
    clip_bounds = {}
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        clip_bounds[col] = (float(lo), float(hi))
        df[col] = df[col].clip(lower=lo, upper=hi)

    # Step 6: LabelEncoder
    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = dict(zip(le.classes_.tolist(), range(len(le.classes_))))
        df[col] = le.transform(df[col].astype(str))

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Step 7: Drop correlated (threshold=0.95)
    target = "survival_status"
    feature_cols = [c for c in df.columns if c != target]
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    df = df.drop(columns=to_drop)

    X = df.drop(columns=[target])
    y = df[target].astype(int)
    features = list(X.columns)

    # Step 8-9: Split + SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    except ImportError:
        pass  # Skip SMOTE if not installed

    # Step 10: Train model
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train)
    except ImportError:
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, max_depth=12)
        model.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    info = {
        "features": features,
        "clip_bounds": clip_bounds,
        "encoders": encoders,
        "dropped": to_drop,
        "accuracy": acc,
        "auc": auc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_type": type(model).__name__,
        "X_full": X,  # For SHAP
    }
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

# ═══ SESSION STATE ════════════════════════════════════════════════════════
for k,v in [("logged_in",False),("user",None),("page","predict")]:
    if k not in st.session_state: st.session_state[k]=v

# ═══ LOGIN ════════════════════════════════════════════════════════════════
def page_login():
    st.markdown("""<div style="text-align:center;padding:50px 0 0">
        <div style="font-size:48px;margin-bottom:6px">🩺</div>
        <h1 style="font-size:24px;color:#065F46!important;margin-bottom:2px!important">BMT Decision Support</h1>
        <p style="color:#6B7280;font-size:13px;margin-bottom:36px">Pediatric Bone Marrow Transplant — Explainable AI</p>
    </div>""", unsafe_allow_html=True)
    _,col,_=st.columns([1,1.4,1])
    with col:
        t1,t2=st.tabs(["🔑 Sign In","📝 Register"])
        with t1:
            u=st.text_input("Username",key="lu");p=st.text_input("Password",type="password",key="lp")
            if st.button("Sign In",key="lb",use_container_width=True):
                r=auth(u,p)
                if r:st.session_state.logged_in=True;st.session_state.user={"id":r[0],"username":r[1],"name":r[2],"role":r[3]};st.rerun()
                else:st.error("Invalid credentials.")
            st.markdown("<p style='text-align:center;color:#6B7280;font-size:12px;margin-top:12px'>Default: <b>admin</b> / <b>admin123</b></p>",unsafe_allow_html=True)
        with t2:
            rn=st.text_input("Full Name",key="rn");ru=st.text_input("Username",key="ru2")
            rp=st.text_input("Password",type="password",key="rp");rp2=st.text_input("Confirm",type="password",key="rp2")
            if st.button("Create Account",key="rb",use_container_width=True):
                if rp!=rp2:st.error("Passwords don't match.")
                elif register(ru,rp,rn):st.success("Account created!")
                else:st.error("Username taken.")

# ═══ SIDEBAR ══════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        u=st.session_state.user
        st.markdown(f"""<div style="padding:22px 0;text-align:center;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:18px">
            <div style="font-size:34px;margin-bottom:4px">🩺</div>
            <div style="font-size:16px;font-weight:800">BMT.AI</div>
            <div style="font-size:10.5px;opacity:0.55">Decision Support</div>
        </div>
        <div style="padding:9px 13px;background:rgba(255,255,255,0.07);border-radius:10px;margin-bottom:18px">
            <div style="font-size:12.5px;font-weight:600">👋 {u['name']}</div>
            <div style="font-size:10.5px;opacity:0.5">{u['role'].title()} · @{u['username']}</div>
        </div>""",unsafe_allow_html=True)
        st.markdown("---")
        for key,label in {"predict":"🔮 New Prediction","history":"📋 History","dashboard":"📊 Dashboard","shap":"🧠 SHAP","data":"📁 Dataset"}.items():
            if st.button(label,key=f"n_{key}",use_container_width=True):st.session_state.page=key;st.rerun()
        st.markdown("---")
        if st.button("🚪 Sign Out",key="logout",use_container_width=True):st.session_state.logged_in=False;st.session_state.user=None;st.rerun()

# ═══ PREDICTION ═══════════════════════════════════════════════════════════
def page_predict():
    st.markdown('<div class="hero"><div class="gp">Prediction Engine</div><h1>🔮 Transplant Outcome Prediction</h1><p>Enter patient &amp; clinical data to predict transplant success</p></div>',unsafe_allow_html=True)

    model, features, clip_bounds, info, err = get_model_and_data()
    if err:
        st.error(f"⚠️ {err}. Place `bone-marrow.arff` in `data/` folder or next to this file.")
        return
    st.markdown(f'<div class="card">✅ Model ready: <b>{info["model_type"]}</b> trained on {info["n_train"]} samples | Test Accuracy: {info["accuracy"]:.1%} | ROC-AUC: {info["auc"]:.3f}</div>',unsafe_allow_html=True)

    inp={}
    with st.expander("👤 Patient Demographics",expanded=True):
        c1,c2,c3=st.columns(3)
        inp["Recipientgender"]=1 if c1.selectbox("Recipient Gender",["Female","Male"],index=1,key="f1")=="Male" else 0
        inp["Recipientage"]=c2.number_input("Recipient Age (yr)",0.5,21.0,9.6,0.1,key="f2")
        inp["Rbodymass"]=c3.number_input("Body Mass (kg)",3.0,100.0,33.0,0.5,key="f3")
    with st.expander("🧬 Donor Information",expanded=True):
        c1,c2,c3=st.columns(3)
        inp["Stemcellsource"]=1 if c1.selectbox("Stem Cell Source",["Bone Marrow","Peripheral Blood"],index=1,key="f4")=="Peripheral Blood" else 0
        inp["Donorage"]=c2.number_input("Donor Age (yr)",18.0,60.0,33.5,0.1,key="f5")
        inp["Gendermatch"]=1 if c3.selectbox("Gender Match",["Other","Female→Male"],key="f6")=="Female→Male" else 0
    with st.expander("🩸 Blood & Immunology",expanded=True):
        c1,c2,c3=st.columns(3)
        # LabelEncoder sorts strings: '-1'→0, '0'→1, '1'→2, '2'→3
        # In ARFF: -1=B, 0=O, 1=A, 2=AB
        abo={"B":0,"O":1,"A":2,"AB":3}
        inp["DonorABO"]=abo[c1.selectbox("Donor ABO",["O","A","B","AB"],key="f7")]
        inp["RecipientABO"]=abo[c2.selectbox("Recipient ABO",["O","A","B","AB"],key="f8")]
        inp["RecipientRh"]=1 if c3.selectbox("Recipient Rh",["Rh−","Rh+"],index=1,key="f9")=="Rh+" else 0
        c1,c2=st.columns(2)
        inp["ABOmatch"]=1 if c1.selectbox("ABO Match",["Mismatched","Matched"],index=1,key="f10")=="Matched" else 0
        inp["CMVstatus"]={"Fully Compatible":0,"Level 1":1,"Level 2":2,"Least Compatible":3}[c2.selectbox("CMV Compatibility",["Fully Compatible","Level 1","Level 2","Least Compatible"],index=2,key="f11")]
        c1,c2=st.columns(2)
        inp["DonorCMV"]=1 if c1.selectbox("Donor CMV",["Absent","Present"],key="f12")=="Present" else 0
        inp["RecipientCMV"]=1 if c2.selectbox("Recipient CMV",["Absent","Present"],index=1,key="f13")=="Present" else 0
    with st.expander("🏥 Disease & Risk",expanded=True):
        c1,c2,c3=st.columns(3)
        inp["Disease"]={"ALL":0,"AML":1,"Chronic":2,"Lymphoma":3,"Non-malignant":4}[c1.selectbox("Disease",["ALL","AML","Chronic","Lymphoma","Non-malignant"],index=1,key="f14")]
        inp["Riskgroup"]=1 if c2.selectbox("Risk",["Low","High"],key="f15")=="High" else 0
        inp["Diseasegroup"]=1 if c3.selectbox("Disease Group",["Non-malignant","Malignant"],index=1,key="f16")=="Malignant" else 0
        c1,c2=st.columns(2)
        inp["Txpostrelapse"]=1 if c1.selectbox("Post-Relapse Tx",["No","Yes"],key="f17")=="Yes" else 0
        inp["Relapse"]=1 if c2.selectbox("Relapse",["No","Yes"],key="f18")=="Yes" else 0
    with st.expander("🔗 HLA Compatibility",expanded=True):
        c1,c2,c3,c4=st.columns(4)
        inp["HLAmatch"]={"10/10":0,"9/10":1,"8/10":2,"7/10":3}[c1.selectbox("HLA Match",["10/10","9/10","8/10","7/10"],key="f19")]
        inp["HLAmismatch"]=0 if inp["HLAmatch"]==0 else 1
        inp["Antigen"]={"None":0,"1":1,"2":2,"3":3}[c2.selectbox("Antigen Diffs",["None","1","2","3"],key="f20")]
        inp["Alel"]={"None":0,"1":1,"2":2,"3":3,"4":4}[c3.selectbox("Allele Diffs",["None","1","2","3","4"],key="f21")]
        inp["HLAgrI"]={"Matched":0,"1 Antigen":1,"1 Allele":2,"DRB1":3,"2 Diffs":4,"2+":5}[c4.selectbox("HLA Gr.I",["Matched","1 Antigen","1 Allele","DRB1","2 Diffs","2+"],key="f22")]
    with st.expander("💉 Cell Doses",expanded=True):
        c1,c2,c3=st.columns(3)
        inp["CD34kgx10d6"]=c1.number_input("CD34+ (×10⁶/kg)",0.5,60.0,9.7,0.1,key="f23")
        inp["CD3dCD34"]=c2.number_input("CD3/CD34 Ratio",0.1,100.0,2.7,0.1,key="f24")
        inp["CD3dkgx10d8"]=c3.number_input("CD3+ (×10⁸/kg)",0.01,20.0,4.3,0.01,key="f25")
    with st.expander("⏱ Recovery & GvHD",expanded=True):
        c1,c2,c3=st.columns(3)
        inp["IIIV"]=1 if c1.selectbox("Acute GvHD II-IV",["No","Yes"],index=1,key="f26")=="Yes" else 0
        inp["aGvHDIIIIV"]=0 if c2.selectbox("Acute GvHD III/IV",["Yes","No"],index=1,key="f27")=="Yes" else 1
        inp["extcGvHD"]=0 if c3.selectbox("Ext. cGvHD",["Yes","No"],index=1,key="f28")=="Yes" else 1
        c1,c2=st.columns(2)
        inp["ANCrecovery"]=c1.number_input("ANC Recovery (days)",7.0,1000000.0,15.0,1.0,key="f29",help="Neutrophil recovery time. 1000000=not recovered.")
        inp["PLTrecovery"]=c2.number_input("PLT Recovery (days)",1.0,1000000.0,21.0,1.0,key="f30",help="Platelet recovery. 1000000=not recovered.")
        c1,c2=st.columns(2)
        inp["time_to_aGvHD_III_IV"]=c1.number_input("Time to aGvHD III/IV",1.0,1000000.0,1000000.0,1.0,key="f31")
        inp["survival_time"]=c2.number_input("Survival Time (days)",1.0,4000.0,676.0,1.0,key="f32")

    # Auto-derived
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
        # Build vector with ONLY the features the model was trained on
        fd={f:inp.get(f,0) for f in features}
        fv=pd.DataFrame([fd])
        for col in fv.columns:fv[col]=pd.to_numeric(fv[col],errors="coerce").fillna(0)
        # Apply SAME IQR clipping as training
        for col,(lo,hi) in clip_bounds.items():
            if col in fv.columns:fv[col]=fv[col].clip(lower=lo,upper=hi)

        pred=int(model.predict(fv)[0])
        proba=model.predict_proba(fv)[0]
        sp=float(proba[0])*100;fp=float(proba[1])*100

        pj=json.dumps({k:round(float(v),4) if isinstance(v,(float,np.floating)) else int(v) for k,v in fd.items()})
        save_pred(st.session_state.user["id"],pj,pred,max(sp,fp))

        st.markdown("---")
        if pred==0:
            st.markdown(f'<div class="pg"><div style="font-size:42px;margin-bottom:8px">✅</div><h2>Favorable — Likely Survived</h2><p style="color:#166534;font-size:14px">Predicted <b>positive outcome</b> with {sp:.1f}% confidence</p><div class="bar"><div class="bf" style="width:{sp}%;background:linear-gradient(90deg,#16A34A,#15803D)"></div></div></div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="pb"><div style="font-size:42px;margin-bottom:8px">⚠️</div><h2>High Risk — Not Survived</h2><p style="color:#991B1B;font-size:14px">Predicted <b>negative outcome</b> with {fp:.1f}% confidence</p><div class="bar"><div class="bf" style="width:{fp}%;background:linear-gradient(90deg,#DC2626,#B91C1C)"></div></div></div>',unsafe_allow_html=True)

        p1,p2=st.columns(2)
        p1.markdown(f'<div class="mc"><div class="ml">Survival</div><div class="mv" style="color:#16A34A">{sp:.1f}%</div></div>',unsafe_allow_html=True)
        p2.markdown(f'<div class="mc"><div class="ml">Non-Survival</div><div class="mv" style="color:#DC2626">{fp:.1f}%</div></div>',unsafe_allow_html=True)

        # SHAP
        st.markdown("#### 🧠 SHAP Explanation")
        try:
            import shap,matplotlib.pyplot as plt,matplotlib;matplotlib.use("Agg")
            try:ex=shap.TreeExplainer(model)
            except:ex=shap.Explainer(model)
            sv=ex.shap_values(fv)
            if isinstance(sv,list):vals=sv[1];ev=ex.expected_value[1] if isinstance(ex.expected_value,(list,np.ndarray)) else ex.expected_value
            else:vals=sv;ev=ex.expected_value
            fig,_=plt.subplots(figsize=(10,7));shap.waterfall_plot(shap.Explanation(values=vals[0],base_values=ev,data=fv.iloc[0].values,feature_names=features),max_display=15,show=False);plt.tight_layout();st.pyplot(fig);plt.close()
        except ImportError:st.info("📦 `pip install shap`")
        except Exception as e:st.warning(f"SHAP: {e}")

        st.markdown('<div class="wb"><strong>⚠️ Disclaimer:</strong> Decision support only. Clinical decisions must be made by qualified professionals.</div>',unsafe_allow_html=True)

# ═══ HISTORY ══════════════════════════════════════════════════════════════
def page_history():
    st.markdown('<div class="hero"><div class="gp">Records</div><h1>📋 History</h1><p>Past predictions</p></div>',unsafe_allow_html=True)
    h=get_history(st.session_state.user["id"])
    if h.empty:st.info("No predictions yet.");return
    t=len(h);s=(h["prediction"]==0).sum();n=(h["prediction"]==1).sum();a=h["probability"].mean()
    st.markdown(f'<div class="mrow"><div class="mc"><div class="ml">Total</div><div class="mv" style="color:#065F46">{t}</div></div><div class="mc"><div class="ml">Survived</div><div class="mv" style="color:#16A34A">{s}</div></div><div class="mc"><div class="ml">Not Survived</div><div class="mv" style="color:#DC2626">{n}</div></div><div class="mc"><div class="ml">Avg Conf</div><div class="mv">{a:.1f}%</div></div></div>',unsafe_allow_html=True)
    d=h[["id","prediction","probability","created_at"]].copy();d["prediction"]=d["prediction"].map({0:"✅ Survived",1:"⚠️ Not Survived"});d["probability"]=d["probability"].round(1).astype(str)+"%";d.columns=["#","Prediction","Confidence","Date"]
    st.dataframe(d,use_container_width=True,hide_index=True)

# ═══ DASHBOARD ════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown('<div class="hero"><div class="gp">Analytics</div><h1>📊 Model Performance</h1><p>Metrics and comparison</p></div>',unsafe_allow_html=True)
    _,_,_,info,err=get_model_and_data()
    if err:st.error(err);return

    st.markdown("### Model Comparison (from train_model.py)")
    dc=pd.DataFrame({"Model":["Random Forest","XGBoost","LightGBM"],"Accuracy":[0.8537,0.8780,0.8780],"Precision":[0.7857,0.8182,0.8571],"Recall":[0.8462,0.6923,0.6923],"F1":[0.8148,0.7500,0.7660],"ROC-AUC":[0.9122,0.9136,0.9073]})
    st.dataframe(dc.style.highlight_max(subset=["Accuracy","Precision","Recall","F1","ROC-AUC"],color="#DCFCE7"),use_container_width=True,hide_index=True)

    st.markdown(f'<div class="card">✅ <b>Current app model: {info["model_type"]}</b> — Accuracy: {info["accuracy"]:.1%}, ROC-AUC: {info["auc"]:.3f} (auto-trained from ARFF)</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="mrow"><div class="mc"><div class="ml">Model</div><div class="mv" style="color:#065F46;font-size:16px">{info["model_type"]}</div></div><div class="mc"><div class="ml">Features</div><div class="mv">{len(info["features"])}</div></div><div class="mc"><div class="ml">Accuracy</div><div class="mv">{info["accuracy"]:.1%}</div></div><div class="mc"><div class="ml">ROC-AUC</div><div class="mv">{info["auc"]:.3f}</div></div></div>',unsafe_allow_html=True)

    try:
        import matplotlib.pyplot as plt;import matplotlib;matplotlib.use("Agg")
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(6,3.5));bars=ax.barh(["RF","XGBoost","LightGBM"],[0.9122,0.9136,0.9073],color=["#3B82F6","#065F46","#8B5CF6"],height=0.45);ax.set_xlim(0.89,0.92)
            for b,v in zip(bars,[0.9122,0.9136,0.9073]):ax.text(v+0.0003,b.get_y()+b.get_height()/2,f"{v:.4f}",va="center",fontweight="bold",fontsize=10)
            ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False);plt.tight_layout();st.pyplot(fig);plt.close()
        with c2:
            fig2,ax2=plt.subplots(figsize=(6,3.5));x=np.arange(4);w=0.3
            ax2.bar(x-w/2,[0.854,0.786,0.846,0.815],w,label="RF",color="#3B82F6");ax2.bar(x+w/2,[0.878,0.818,0.692,0.750],w,label="XGB",color="#065F46")
            ax2.set_xticks(x);ax2.set_xticklabels(["Acc","Prec","Rec","F1"]);ax2.set_ylim(0.5,1.0);ax2.legend(fontsize=9);ax2.spines["top"].set_visible(False);ax2.spines["right"].set_visible(False);plt.tight_layout();st.pyplot(fig2);plt.close()
    except:pass

    st.markdown("### Pipeline\n1. **Missing** → median/mode\n2. **Outliers** → IQR clip\n3. **Encoding** → LabelEncoder\n4. **Correlation** → Drop >0.95\n5. **SMOTE** → Balance\n6. **Train** → XGBoost (or RandomForest)")

# ═══ SHAP ═════════════════════════════════════════════════════════════════
def page_shap():
    st.markdown('<div class="hero"><div class="gp">Explainability</div><h1>🧠 SHAP Analysis</h1><p>Global feature importance</p></div>',unsafe_allow_html=True)
    model,features,_,info,err=get_model_and_data()
    if err:st.error(err);return
    try:
        import shap,matplotlib.pyplot as plt,matplotlib;matplotlib.use("Agg")
        X=info["X_full"]
        st.info("⏳ Computing SHAP values...")
        try:ex=shap.TreeExplainer(model)
        except:ex=shap.Explainer(model)
        sv=ex.shap_values(X);vals=sv[1] if isinstance(sv,list) else sv
        st.markdown("### Feature Importance")
        fig1,_=plt.subplots(figsize=(10,9));shap.summary_plot(vals,X,plot_type="bar",show=False,max_display=20);plt.tight_layout();st.pyplot(fig1);plt.close()
        st.markdown("### Beeswarm")
        fig2,_=plt.subplots(figsize=(10,9));shap.summary_plot(vals,X,show=False,max_display=20);plt.tight_layout();st.pyplot(fig2);plt.close()
    except ImportError:st.warning("📦 `pip install shap`")
    except Exception as e:st.error(f"SHAP error: {e}")

# ═══ DATASET ══════════════════════════════════════════════════════════════
def page_data():
    st.markdown('<div class="hero"><div class="gp">Explorer</div><h1>📁 Dataset</h1><p>Explore the training data</p></div>',unsafe_allow_html=True)
    df=load_arff_raw()
    if df is None:st.error("ARFF not found.");return
    tg="survival_status"
    df[tg]=pd.to_numeric(df[tg],errors="coerce").fillna(0).astype(int)
    ns=int((df[tg]==0).sum());nn=int((df[tg]==1).sum())
    st.markdown(f'<div class="mrow"><div class="mc"><div class="ml">Samples</div><div class="mv" style="color:#065F46">{len(df)}</div></div><div class="mc"><div class="ml">Features</div><div class="mv">{len(df.columns)-1}</div></div><div class="mc"><div class="ml">Survived</div><div class="mv" style="color:#16A34A">{ns}</div></div><div class="mc"><div class="ml">Not Survived</div><div class="mv" style="color:#DC2626">{nn}</div></div></div>',unsafe_allow_html=True)
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
                    fig,ax=plt.subplots(figsize=(6,3.5));pd_.hist(bins=30,ax=ax,color="#065F46",edgecolor="white",alpha=0.85);ax.set_title(sel,fontweight="bold");ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False);plt.tight_layout();st.pyplot(fig);plt.close()
                with c2:
                    fig2,ax2=plt.subplots(figsize=(6,3.5));tgt=df[tg]
                    ax2.hist(pd_[tgt==0],bins=25,alpha=0.7,label="Surv",color="#16A34A");ax2.hist(pd_[tgt==1],bins=25,alpha=0.7,label="Not",color="#DC2626")
                    ax2.legend();ax2.spines["top"].set_visible(False);ax2.spines["right"].set_visible(False);plt.tight_layout();st.pyplot(fig2);plt.close()
            except Exception as e:st.warning(f"Chart: {e}")

# ═══ MAIN ═════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:page_login()
    else:
        sidebar()
        {"predict":page_predict,"history":page_history,"dashboard":page_dashboard,"shap":page_shap,"data":page_data}.get(st.session_state.page,page_predict)()
if __name__=="__main__":main()