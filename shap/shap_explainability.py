"""
src/shap_explainability.py
SHAP Explainability Module — Bone Marrow Transplant Prediction
Compatible with data_processing.py and train_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib


# ─────────────────────────────────────────────
# 1. CHARGEMENT DU MODÈLE ET DES DONNÉES
# ─────────────────────────────────────────────

def load_model(model_path: str):
    """Charge un modèle sauvegardé avec joblib."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    model = joblib.load(model_path)
    print(f"✅ Modèle chargé depuis : {model_path}")
    return model


# ─────────────────────────────────────────────
# 2. CRÉATION DE L'EXPLAINER SHAP
# ─────────────────────────────────────────────

def build_shap_explainer(model, X_train: pd.DataFrame):
    """
    Crée l'explainer SHAP adapté au type de modèle.
    - TreeExplainer  → Random Forest, XGBoost, LightGBM (rapide, exact)
    - KernelExplainer → SVM ou modèles non-arborescents (lent, approché)
    """
    model_type = type(model).__name__

    tree_based = ["RandomForestClassifier", "XGBClassifier", "LGBMClassifier",
                  "GradientBoostingClassifier", "ExtraTreesClassifier"]

    if model_type in tree_based:
        print(f"🌲 TreeExplainer sélectionné pour : {model_type}")
        explainer = shap.TreeExplainer(model)
    else:
        print(f"⚙️  KernelExplainer sélectionné pour : {model_type}")
        # Résumé du background pour accélérer KernelExplainer
        background = shap.kmeans(X_train, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)

    return explainer


# ─────────────────────────────────────────────
# 3. CALCUL DES VALEURS SHAP
# ─────────────────────────────────────────────

def compute_shap_values(explainer, X: pd.DataFrame):
    """
    Calcule les valeurs SHAP.
    Retourne un array 2D (n_samples × n_features) pour la classe positive (1).
    """
    print("⏳ Calcul des valeurs SHAP en cours...")
    shap_values = explainer.shap_values(X)

    # Pour les classifieurs binaires, shap_values peut être une liste [classe_0, classe_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]   # classe positive = survie

    print(f"✅ Valeurs SHAP calculées — shape : {np.array(shap_values).shape}")
    return shap_values


# ─────────────────────────────────────────────
# 4. VISUALISATIONS SHAP
# ─────────────────────────────────────────────

def plot_summary(shap_values, X: pd.DataFrame, output_dir: str = "outputs/shap"):
    """
    Beeswarm plot : importance + direction d'effet de chaque feature.
    C'est la visualisation principale demandée dans le projet.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"📊 Summary plot (beeswarm) sauvegardé → {path}")
    return path


def plot_bar_importance(shap_values, X: pd.DataFrame, output_dir: str = "outputs/shap"):
    """
    Bar plot : importance moyenne absolue de chaque feature (top 20).
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    path = os.path.join(output_dir, "shap_feature_importance_bar.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"📊 Bar importance plot sauvegardé → {path}")
    return path


def plot_waterfall_single(explainer, X: pd.DataFrame,
                          sample_index: int = 0,
                          output_dir: str = "outputs/shap"):
    """
    Waterfall plot pour UN patient donné.
    Montre la contribution de chaque feature à la prédiction individuelle.
    Utilisé dans l'interface Streamlit pour expliquer chaque cas.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Utilisation de l'API Explanation (SHAP >= 0.40)
    explanation = explainer(X.iloc[[sample_index]])

    # Classe positive
    if len(explanation.shape) == 3:
        explanation = explanation[:, :, 1]

    plt.figure()
    shap.waterfall_plot(explanation[0], show=False)
    path = os.path.join(output_dir, f"shap_waterfall_patient_{sample_index}.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"📊 Waterfall plot (patient {sample_index}) sauvegardé → {path}")
    return path


def plot_force_single(explainer, shap_values, X: pd.DataFrame,
                      sample_index: int = 0,
                      output_dir: str = "outputs/shap"):
    """
    Force plot pour UN patient (version statique PNG).
    Alternative au waterfall, plus compacte — utile dans l'interface.
    """
    os.makedirs(output_dir, exist_ok=True)
    shap.initjs()

    force = shap.force_plot(
        explainer.expected_value if not isinstance(explainer.expected_value, list)
        else explainer.expected_value[1],
        shap_values[sample_index],
        X.iloc[sample_index],
        matplotlib=True,
        show=False
    )
    path = os.path.join(output_dir, f"shap_force_patient_{sample_index}.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"📊 Force plot (patient {sample_index}) sauvegardé → {path}")
    return path


# ─────────────────────────────────────────────
# 5. RÉSUMÉ TEXTUEL DES FEATURES IMPORTANTES
# ─────────────────────────────────────────────

def get_top_features(shap_values, X: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Retourne un DataFrame trié par importance SHAP moyenne absolue.
    Utilisé pour répondre à la question du README :
    'Which medical features most influenced predictions?'
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df_importance = pd.DataFrame({
        "Feature":    X.columns,
        "SHAP_Mean_Abs": mean_abs
    }).sort_values("SHAP_Mean_Abs", ascending=False).head(n).reset_index(drop=True)

    df_importance.index += 1   # rank starts at 1
    print(f"\n🔍 Top {n} features les plus influentes :")
    print(df_importance.to_string())
    return df_importance


# ─────────────────────────────────────────────
# 6. EXPLICATION D'UN PATIENT UNIQUE (pour Streamlit)
# ─────────────────────────────────────────────

def explain_single_patient(explainer, model, patient_df: pd.DataFrame) -> dict:
    """
    Entrée  : DataFrame d'UNE seule ligne (données patient depuis l'interface).
    Sortie  : dict avec probabilité de survie + contributions SHAP triées.

    Utilisé directement dans app/app.py pour afficher l'explication au médecin.
    """
    # Prédiction
    proba_survival = model.predict_proba(patient_df)[0][1]

    # Valeurs SHAP
    sv = explainer.shap_values(patient_df)
    if isinstance(sv, list):
        sv = sv[1]

    contributions = pd.DataFrame({
        "Feature":      patient_df.columns,
        "Valeur":       patient_df.iloc[0].values,
        "SHAP":         sv[0]
    }).sort_values("SHAP", key=abs, ascending=False).reset_index(drop=True)

    return {
        "proba_survival":  round(float(proba_survival), 4),
        "prediction":      int(proba_survival >= 0.5),
        "contributions":   contributions
    }


# ─────────────────────────────────────────────
# 7. SAUVEGARDE DE L'EXPLAINER
# ─────────────────────────────────────────────

def save_explainer(explainer, output_dir: str = "models"):
    """Sauvegarde l'explainer pour réutilisation dans l'interface."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "shap_explainer.pkl")
    joblib.dump(explainer, path)
    print(f"💾 Explainer SHAP sauvegardé → {path}")
    return path


def load_explainer(path: str = "models/shap_explainer.pkl"):
    """Charge l'explainer depuis le disque."""
    explainer = joblib.load(path)
    print(f"✅ Explainer SHAP chargé depuis : {path}")
    return explainer


# ─────────────────────────────────────────────
# 8. PIPELINE COMPLET
# ─────────────────────────────────────────────

def run_shap_pipeline(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      output_dir: str = "outputs/shap",
                      save_dir: str = "models"):
    """
    Pipeline SHAP complet :
      1. Crée l'explainer
      2. Calcule les valeurs SHAP sur X_test
      3. Génère tous les plots
      4. Affiche le classement des features
      5. Sauvegarde l'explainer

    Retourne : (explainer, shap_values, top_features_df)
    """
    print("\n" + "="*50)
    print("  🔍 SHAP EXPLAINABILITY PIPELINE")
    print("="*50)

    explainer   = build_shap_explainer(model, X_train)
    shap_values = compute_shap_values(explainer, X_test)

    # Plots globaux
    plot_summary(shap_values, X_test, output_dir)
    plot_bar_importance(shap_values, X_test, output_dir)

    # Plots individuels (patient 0 et 1 à titre d'exemple)
    for idx in range(min(2, len(X_test))):
        try:
            plot_waterfall_single(explainer, X_test, sample_index=idx, output_dir=output_dir)
        except Exception as e:
            print(f"⚠️  Waterfall plot patient {idx} ignoré : {e}")

    # Top features
    top_features = get_top_features(shap_values, X_test, n=10)

    # Sauvegarde
    save_explainer(explainer, save_dir)

    return explainer, shap_values, top_features


# ─────────────────────────────────────────────
# POINT D'ENTRÉE STANDALONE
# ─────────────────────────────────────────────
'''
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_processing import preprocess_pipeline

    data_path  = sys.argv[1] if len(sys.argv) > 1 else "data/bone-marrow.arff"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/random_forest.pkl"

    print("🔄 Chargement des données...")
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path)

    print("🤖 Chargement du modèle...")
    model = load_model(model_path)

    explainer, shap_values, top_features = run_shap_pipeline(
        model, X_train, X_test,
        output_dir="outputs/shap",
        save_dir="models"
    )

    print("\n✅ SHAP pipeline terminé avec succès !")'''
if __name__ == "__main__":
    import sys

    # Ajoute src/ au path depuis n'importe où
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_root, "src"))

    from data_processing import preprocess_pipeline

    data_path  = sys.argv[1] if len(sys.argv) > 1 else "data/bone-marrow.arff"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/xgboost.pkl"

    print("🔄 Chargement des données...")
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path)

    print("🤖 Chargement du modèle...")
    model = load_model(model_path)

    explainer, shap_values, top_features = run_shap_pipeline(
        model, X_train, X_test,
        output_dir="outputs/shap",
        save_dir="models"
    )
    print("\n✅ SHAP pipeline terminé avec succès !")
    
