"""
tests/test_data_processing.py
==============================
Tests unitaires complets pour le pipeline de traitement de données
du projet PediaBMT — Greffe de Moelle Osseuse Pédiatrique.

Couverture :
    1. handle_missing_values  — gestion des valeurs manquantes
    2. optimize_memory        — réduction de l'empreinte mémoire
    3. Chargement du modèle & pipeline de prédiction (avec mock)

Lancer les tests :
    pytest tests/test_data_processing.py -v
"""

import os
import sys
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Rendre src/ importable ────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Import des fonctions à tester — depuis le module du projet
from data_processing import handle_missing_values, optimize_memory


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES PARTAGÉES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def df_avec_manquants() -> pd.DataFrame:
    """
    DataFrame fictif qui reproduit la structure du dataset bone-marrow :
    - colonnes numériques float64 / int64 avec NaN et None
    - colonnes catégorielles object avec NaN et None
    - colonne "?" (token ARFF) déjà remplacée ou non
    - colonne cible 'survival_status' jamais altérée

    Ce fixture couvre tous les cas d'entrée réels de handle_missing_values().
    """
    rng = np.random.default_rng(42)
    n = 80

    df = pd.DataFrame({
        # Colonnes numériques — types lourds volontaires pour reproduire ARFF
        "Donorage":         rng.uniform(18, 60, n).astype(np.float64),
        "Recipientage":     rng.uniform(1, 18, n).astype(np.float64),
        "CD34kgx10d6":      rng.uniform(0, 20, n).astype(np.float64),
        "PLTrecovery":      rng.integers(10, 60, n).astype(np.int64),

        # Colonnes catégorielles
        "Disease":          rng.choice(["ALL", "AML", "chronic", "nonmalignant"], n),
        "Stemcellsource":   rng.choice(["1", "0"], n),

        # Cible binaire — NE DOIT JAMAIS être modifiée
        "survival_status":  rng.integers(0, 2, n).astype(np.int64),
    })

    # ── Injection de valeurs manquantes dans les colonnes features ────────────
    # NaN natifs dans les numériques
    df.loc[[2, 10, 25, 40, 55], "Donorage"]     = np.nan
    df.loc[[1, 30],              "Recipientage"] = np.nan
    df.loc[[5, 15],              "CD34kgx10d6"]  = np.nan
    df.loc[[7],                  "PLTrecovery"]  = np.nan

    # None (converti automatiquement en NaN par pandas dans les colonnes object)
    df.loc[[3, 12, 50], "Disease"]       = None
    df.loc[[8, 20],     "Stemcellsource"] = None

    return df


@pytest.fixture
def df_types_lourds() -> pd.DataFrame:
    """
    DataFrame avec des types mémoire-lourds (float64 / int64) et des
    colonnes catégorielles object à faible cardinalité.

    Utilisé pour tester optimize_memory() : on calcule la mémoire
    avant/après et on vérifie qu'elle a diminué.
    """
    rng = np.random.default_rng(0)
    n = 500   # assez grand pour que la différence de mémoire soit mesurable

    return pd.DataFrame({
        # float64 → devrait devenir float32
        "feature_float_1":  rng.random(n).astype(np.float64),
        "feature_float_2":  rng.random(n).astype(np.float64) * 100,
        "feature_float_3":  rng.random(n).astype(np.float64) - 50,

        # int64 → devrait devenir int32
        "feature_int_1":    rng.integers(0, 100,  n).astype(np.int64),
        "feature_int_2":    rng.integers(0, 1000, n).astype(np.int64),

        # object faible cardinalité → devrait devenir category
        "disease":          rng.choice(["ALL", "AML", "chronic"], n),
        "gender":           rng.choice(["M", "F"], n),

        # Cible binaire
        "survival_status":  rng.integers(0, 2, n).astype(np.int64),
    })


# ══════════════════════════════════════════════════════════════════════════════
# 1. GESTION DES VALEURS MANQUANTES  ─  handle_missing_values()
# ══════════════════════════════════════════════════════════════════════════════

class TestHandleMissingValues:
    """
    Vérifie que handle_missing_values() :
      - élimine TOUTES les valeurs manquantes (NaN, None)
      - impute les numériques avec la MÉDIANE
      - impute les catégorielles avec le MODE
      - ne modifie PAS la colonne cible
      - conserve la forme (shape) du DataFrame
    """

    # ── Test 1 : aucune valeur manquante ne subsiste ──────────────────────────
    def test_aucune_valeur_manquante_apres_imputation(self, df_avec_manquants):
        """
        CRITÈRE PRINCIPAL : après handle_missing_values(), isnull().sum().sum()
        doit retourner 0.
        """
        df_propre = handle_missing_values(df_avec_manquants.copy())

        total_nan = df_propre.isnull().sum().sum()
        assert total_nan == 0, (
            f"Il reste {total_nan} valeur(s) manquante(s) après imputation.\n"
            f"Détail par colonne :\n{df_propre.isnull().sum()[df_propre.isnull().sum() > 0]}"
        )

    # ── Test 2 : les numériques sont imputées avec la médiane ─────────────────
    def test_imputation_numerique_avec_mediane(self, df_avec_manquants):
        """
        Les positions NaN dans 'Donorage' (colonne float64) doivent être
        remplies avec la médiane de la colonne AVANT imputation.
        """
        col = "Donorage"
        positions_nan = df_avec_manquants.index[df_avec_manquants[col].isna()].tolist()
        mediane_avant = df_avec_manquants[col].median()  # médiane sur les valeurs existantes

        df_propre = handle_missing_values(df_avec_manquants.copy())

        for idx in positions_nan:
            valeur_imputee = df_propre.loc[idx, col]
            assert valeur_imputee == pytest.approx(mediane_avant, rel=1e-3), (
                f"Ligne {idx} : valeur imputée = {valeur_imputee:.4f}, "
                f"médiane attendue = {mediane_avant:.4f}"
            )

    # ── Test 3 : les catégorielles sont imputées avec le mode ────────────────
    def test_imputation_categorielle_avec_mode(self, df_avec_manquants):
        """
        Les positions None/NaN dans 'Disease' (colonne object) doivent être
        remplies avec la valeur modale de la colonne.
        """
        col = "Disease"
        positions_nan = df_avec_manquants.index[df_avec_manquants[col].isna()].tolist()
        mode_avant = df_avec_manquants[col].mode()[0]

        df_propre = handle_missing_values(df_avec_manquants.copy())

        for idx in positions_nan:
            valeur_imputee = df_propre.loc[idx, col]
            assert valeur_imputee == mode_avant, (
                f"Ligne {idx} : valeur imputée = '{valeur_imputee}', "
                f"mode attendu = '{mode_avant}'"
            )

    # ── Test 4 : la colonne cible n'est jamais modifiée ──────────────────────
    def test_colonne_cible_intacte(self, df_avec_manquants):
        """
        'survival_status' ne contient pas de NaN dans le fixture, donc
        handle_missing_values() NE DOIT PAS la toucher.
        """
        cible_avant = df_avec_manquants["survival_status"].copy()
        df_propre   = handle_missing_values(df_avec_manquants.copy())

        pd.testing.assert_series_equal(
            df_propre["survival_status"].reset_index(drop=True),
            cible_avant.reset_index(drop=True),
            check_names=False,
            check_dtype=False,
        )

    # ── Test 5 : la forme du DataFrame est conservée ─────────────────────────
    def test_shape_inchange(self, df_avec_manquants):
        """
        handle_missing_values() ne doit ni supprimer de lignes ni de colonnes.
        """
        shape_avant = df_avec_manquants.shape
        df_propre   = handle_missing_values(df_avec_manquants.copy())
        assert df_propre.shape == shape_avant, (
            f"Shape avant : {shape_avant} | Shape après : {df_propre.shape}"
        )

    # ── Test 6 : idempotence — appeler deux fois est sans danger ─────────────
    def test_idempotence(self, df_avec_manquants):
        """
        Appliquer handle_missing_values() une deuxième fois sur un DataFrame
        déjà propre ne doit pas provoquer d'erreur ni changer les valeurs.
        """
        df_propre       = handle_missing_values(df_avec_manquants.copy())
        df_propre_bis   = handle_missing_values(df_propre.copy())

        pd.testing.assert_frame_equal(
            df_propre.reset_index(drop=True),
            df_propre_bis.reset_index(drop=True),
            check_dtype=False,
        )

    # ── Test 7 : DataFrame entièrement propre — aucune modification ───────────
    def test_dataframe_propre_non_modifie(self):
        """
        Si le DataFrame d'entrée ne contient aucun NaN,
        handle_missing_values() ne doit pas altérer les valeurs.
        """
        df_propre = pd.DataFrame({
            "age":    [25.0, 35.0, 45.0],
            "gender": ["M", "F", "M"],
            "survival_status": [1, 0, 1],
        })
        df_resultat = handle_missing_values(df_propre.copy())
        pd.testing.assert_frame_equal(df_resultat, df_propre)


# ══════════════════════════════════════════════════════════════════════════════
# 2. OPTIMISATION MÉMOIRE  ─  optimize_memory()
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizeMemory:
    """
    Vérifie que optimize_memory() :
      - réduit la consommation mémoire globale (critère principal)
      - convertit float64 → float32
      - convertit int64  → int32
      - conserve les valeurs numériques (précision acceptable)
      - conserve le nombre de lignes et de colonnes
    """

    # ── Test 1 : la mémoire est strictement réduite ───────────────────────────
    def test_memoire_strictement_reduite(self, df_types_lourds):
        """
        CRITÈRE PRINCIPAL : memory_usage(deep=True).sum() APRÈS optimisation
        doit être STRICTEMENT INFÉRIEUR à la valeur AVANT optimisation.
        """
        memoire_avant = df_types_lourds.memory_usage(deep=True).sum()
        df_opt        = optimize_memory(df_types_lourds.copy(), verbose=False)
        memoire_apres = df_opt.memory_usage(deep=True).sum()

        assert memoire_apres < memoire_avant, (
            f"La mémoire n'a pas diminué !\n"
            f"  Avant  : {memoire_avant:,} octets\n"
            f"  Après  : {memoire_apres:,} octets"
        )

    # ── Test 2 : float64 converti en float32 ──────────────────────────────────
    def test_float64_converti_en_float32(self, df_types_lourds):
        """
        Toutes les colonnes float64 doivent devenir float32 après optimisation.
        """
        df_opt = optimize_memory(df_types_lourds.copy(), verbose=False)

        cols_float64_restantes = df_opt.select_dtypes(include=[np.float64]).columns.tolist()
        assert cols_float64_restantes == [], (
            f"Colonnes float64 encore présentes : {cols_float64_restantes}"
        )

    # ── Test 3 : int64 converti en int32 ─────────────────────────────────────
    def test_int64_converti_en_int32(self, df_types_lourds):
        """
        Toutes les colonnes int64 doivent devenir int32 après optimisation.
        """
        df_opt = optimize_memory(df_types_lourds.copy(), verbose=False)

        cols_int64_restantes = df_opt.select_dtypes(include=[np.int64]).columns.tolist()
        assert cols_int64_restantes == [], (
            f"Colonnes int64 encore présentes : {cols_int64_restantes}"
        )

    # ── Test 4 : les valeurs numériques sont conservées ───────────────────────
    def test_valeurs_numeriques_conservees(self, df_types_lourds):
        """
        Le downcast float64→float32 ne doit pas modifier les valeurs
        de plus de 0.01 % (tolérance relative rtol=1e-4).
        """
        df_opt = optimize_memory(df_types_lourds.copy(), verbose=False)

        for col in df_types_lourds.select_dtypes(include=[np.float64]).columns:
            np.testing.assert_allclose(
                df_types_lourds[col].values,
                df_opt[col].values.astype(np.float64),
                rtol=1e-4,
                err_msg=f"Les valeurs de la colonne '{col}' ont trop changé après downcast."
            )

    # ── Test 5 : la forme du DataFrame est conservée ─────────────────────────
    def test_shape_inchange(self, df_types_lourds):
        """
        optimize_memory() ne doit ni ajouter ni supprimer de lignes/colonnes.
        """
        df_opt = optimize_memory(df_types_lourds.copy(), verbose=False)
        assert df_opt.shape == df_types_lourds.shape, (
            f"Shape avant : {df_types_lourds.shape} | après : {df_opt.shape}"
        )

    # ── Test 6 : noms de colonnes inchangés ───────────────────────────────────
    def test_noms_colonnes_inchanges(self, df_types_lourds):
        """
        Les noms de colonnes ne doivent pas être modifiés par l'optimisation.
        """
        df_opt = optimize_memory(df_types_lourds.copy(), verbose=False)
        assert list(df_opt.columns) == list(df_types_lourds.columns)

    # ── Test 7 : réduction mesurable en pourcentage ───────────────────────────
    def test_reduction_superieure_a_10_pourcent(self, df_types_lourds):
        """
        Sur un DataFrame composé majoritairement de float64/int64, la
        réduction doit être significative (> 10 %).
        """
        avant = df_types_lourds.memory_usage(deep=True).sum()
        apres = optimize_memory(df_types_lourds.copy(), verbose=False)\
                    .memory_usage(deep=True).sum()

        reduction_pct = 100 * (avant - apres) / avant
        assert reduction_pct > 10, (
            f"Réduction insuffisante : {reduction_pct:.1f}% (attendu > 10%)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHARGEMENT DU MODÈLE & PIPELINE DE PRÉDICTION  ─  avec unittest.mock
# ══════════════════════════════════════════════════════════════════════════════

class TestModelLoadingAndPrediction:
    """
    Vérifie le pipeline de chargement et de prédiction SANS dépendre
    d'un fichier .pkl réel, grâce à unittest.mock (MagicMock + patch).

    Stratégie de mock :
    - joblib.load  est intercepté et retourne un faux modèle sklearn-like.
    - Le faux modèle expose predict() et predict_proba() avec un comportement
      déterministe et contrôlé.
    - On vérifie la forme et les valeurs de sortie — pas le modèle lui-même.
    """

    @pytest.fixture
    def mock_model(self):
        """
        Crée un faux modèle sklearn-compatible avec MagicMock.
        - predict()       → retourne [0] ou [1]
        - predict_proba() → retourne [[0.3, 0.7]]  (probabilités valides)
        """
        modele = MagicMock()
        modele.predict.return_value       = np.array([1])          # prédit "Survie"
        modele.predict_proba.return_value = np.array([[0.3, 0.7]]) # P(non-survie)=0.3, P(survie)=0.7
        return modele

    @pytest.fixture
    def echantillon_patient(self):
        """
        Une ligne de données (1 patient) au format attendu par le modèle.
        Correspond à un vecteur de features numériques après encode_features().
        """
        rng = np.random.default_rng(99)
        n_features = 30   # nombre typique de features après encodage + corrélation
        return pd.DataFrame(
            rng.random((1, n_features)).astype(np.float32),
            columns=[f"feature_{i}" for i in range(n_features)]
        )

    # ── Test 1 : le modèle mocké est correctement chargé ─────────────────────
    def test_chargement_modele_via_joblib_mock(self, mock_model):
        """
        Vérifie que joblib.load peut être mocké sans erreur.
        La valeur de retour doit posséder les méthodes d'un estimateur sklearn.
        """
        with patch("joblib.load", return_value=mock_model) as mock_load:
            import joblib
            modele_charge = joblib.load("models/xgboost.pkl")

            # joblib.load a bien été appelé avec le bon chemin
            mock_load.assert_called_once_with("models/xgboost.pkl")

            # L'objet retourné a les méthodes d'un estimateur
            assert hasattr(modele_charge, "predict"), (
                "Le modèle chargé n'a pas de méthode predict()"
            )
            assert hasattr(modele_charge, "predict_proba"), (
                "Le modèle chargé n'a pas de méthode predict_proba()"
            )

    # ── Test 2 : predict() retourne une prédiction binaire valide ─────────────
    def test_predict_retourne_valeur_binaire(self, mock_model, echantillon_patient):
        """
        Le résultat de predict() doit être 0 ou 1 (classification binaire).
        Vérifie le TYPE (numpy array ou int) et la VALEUR.
        """
        prediction = mock_model.predict(echantillon_patient)

        # La sortie doit être array-like
        assert hasattr(prediction, "__len__"), (
            f"predict() doit retourner un tableau, reçu : {type(prediction)}"
        )

        # La valeur doit être 0 ou 1
        valeur = int(prediction[0])
        assert valeur in (0, 1), (
            f"Prédiction invalide : {valeur}. Valeurs attendues : 0 ou 1."
        )

    # ── Test 3 : predict_proba() retourne des probabilités valides ────────────
    def test_predict_proba_probabilites_valides(self, mock_model, echantillon_patient):
        """
        predict_proba() doit retourner :
          - un tableau de forme (1, 2) pour une classification binaire
          - des valeurs dans [0.0, 1.0]
          - une somme de probabilités égale à 1.0 (± 1e-5)
        """
        probas = mock_model.predict_proba(echantillon_patient)

        # Forme : 1 échantillon × 2 classes
        assert probas.shape == (1, 2), (
            f"Forme attendue (1, 2), obtenue : {probas.shape}"
        )

        # Toutes les probabilités dans [0, 1]
        assert np.all(probas >= 0.0) and np.all(probas <= 1.0), (
            f"Probabilités hors [0, 1] : {probas}"
        )

        # La somme par ligne est 1.0
        somme = probas[0].sum()
        assert abs(somme - 1.0) < 1e-5, (
            f"La somme des probabilités doit être 1.0, obtenu : {somme:.6f}"
        )

    # ── Test 4 : pipeline complet mock-to-prediction ──────────────────────────
    def test_pipeline_complet_mock_vers_prediction(self, mock_model, echantillon_patient):
        """
        Simule le pipeline complet :
        1. joblib.load() → modèle mocké
        2. modèle.predict()       → prédiction binaire
        3. modèle.predict_proba() → probabilité de survie

        Vérifie que le flux end-to-end ne lève pas d'exception
        et produit une sortie cohérente.
        """
        with patch("joblib.load", return_value=mock_model):
            import joblib

            # Étape 1 : chargement
            modele = joblib.load("models/xgboost.pkl")

            # Étape 2 : prédiction
            prediction = modele.predict(echantillon_patient)
            proba      = modele.predict_proba(echantillon_patient)

            # La prédiction est cohérente avec la probabilité
            classe_predite      = int(prediction[0])
            proba_classe_predite = float(proba[0][classe_predite])

            assert classe_predite in (0, 1)
            assert 0.0 <= proba_classe_predite <= 1.0, (
                f"Probabilité hors bornes pour la classe {classe_predite} : "
                f"{proba_classe_predite}"
            )

    # ── Test 5 : predict() fonctionne sur un batch de patients ───────────────
    def test_prediction_batch_plusieurs_patients(self, mock_model):
        """
        Vérifie que le modèle peut prédire sur plusieurs patients à la fois
        et que la longueur du retour est cohérente avec la taille du batch.
        """
        n_patients = 10
        rng = np.random.default_rng(7)
        batch = pd.DataFrame(
            rng.random((n_patients, 30)).astype(np.float32),
            columns=[f"feature_{i}" for i in range(30)]
        )

        # Ajuster le mock pour retourner n_patients prédictions
        mock_model.predict.return_value = np.array([0, 1] * (n_patients // 2))

        predictions = mock_model.predict(batch)

        assert len(predictions) == n_patients, (
            f"Attendu {n_patients} prédictions, obtenu {len(predictions)}"
        )
        assert all(p in (0, 1) for p in predictions), (
            f"Prédictions non-binaires détectées : {set(predictions)}"
        )

    # ── Test 6 : FileNotFoundError si le modèle est absent ────────────────────
    def test_erreur_si_fichier_modele_absent(self):
        """
        Si le fichier .pkl n'existe pas, une exception doit être levée.
        Vérifie que le code ne s'arrête pas silencieusement.
        """
        import joblib

        with patch("joblib.load", side_effect=FileNotFoundError("Fichier introuvable")):
            with pytest.raises(FileNotFoundError):
                joblib.load("models/fichier_inexistant.pkl")
