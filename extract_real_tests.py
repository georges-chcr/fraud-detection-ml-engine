"""
Génère 3 vraies fraudes + 3 transactions normales depuis X_test.pkl / y_test.pkl.
Imprime un tableau JSON de 30 valeurs prêt à coller dans l'UI pour chaque cas.

Ordre des features dans le JSON (identique à l'ordre CSV / attendu par l'API) :
  [0]     Time    — secondes depuis début du dataset (brut, non scalé)
  [1-28]  V1–V28  — composantes PCA (issues du dataset, non scalées davantage)
  [29]    Amount  — montant brut en $ (l'API applique le RobustScaler en interne)

Pourquoi inverse_transform sur Amount :
  X_test.pkl a été créé après application du RobustScaler sur Amount.
  L'API rescale Amount à la volée avant de passer au modèle.
  → on doit donc fournir le montant brut, pas la valeur déjà scalée.
"""
import json
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# ── Chargement des artifacts ──────────────────────────────────────────────────
print("Chargement des artifacts...")
X_test = joblib.load(BASE_DIR / "X_test.pkl")
y_test = joblib.load(BASE_DIR / "y_test.pkl")
model  = joblib.load(BASE_DIR / "models" / "random_forest_fraud_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "rob_scaler.pkl")

X_arr = X_test.values if hasattr(X_test, "values") else np.array(X_test)
y_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)

# ── Probabilités sur tout X_test (Amount déjà scalé dans X_test) ─────────────
print(f"Calcul des probabilités sur {len(X_arr)} transactions...")
probas = model.predict_proba(X_arr)[:, 1]

fraud_idx  = np.where(y_arr == 1)[0]
normal_idx = np.where(y_arr == 0)[0]

# 3 fraudes avec les scores les plus élevés
top3_fraud  = fraud_idx[np.argsort(probas[fraud_idx])[::-1][:3]]
# 3 normales avec les scores les plus bas (transactions les plus "sûres")
top3_normal = normal_idx[np.argsort(probas[normal_idx])[:3]]


def build_api_features(row: np.ndarray) -> list[float]:
    """
    Reconstruit le vecteur à envoyer à l'API :
    - index 0-28 : tels quels depuis X_test (Time + V1-V28 déjà PCA/bruts)
    - index 29   : Amount inverse-transformé → valeur brute en $
    """
    result = row.tolist()
    raw_amount = float(scaler.inverse_transform([[row[29]]])[0][0])
    result[29] = raw_amount
    return result


def print_case(idx: int, label: str, num: int) -> None:
    row      = X_arr[idx]
    features = build_api_features(row)
    prob     = probas[idx]
    amount   = features[29]
    decision = "REJECTED" if prob >= 0.80 else "APPROVED"

    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  {label} #{num}")
    print(f"  Probabilité de fraude attendue : {prob * 100:.2f}%  →  {decision}")
    print(f"  Montant réel                   : ${amount:.2f}")
    print(sep)
    print(json.dumps([round(v, 6) for v in features]))


# ── Affichage ─────────────────────────────────────────────────────────────────
banner = "█" * 72
print(f"\n{banner}")
print("  EXTRACT REAL TESTS — 3 FRAUDES + 3 TRANSACTIONS NORMALES")
print("  Copiez chaque ligne JSON directement dans le champ de l'UI")
print(f"{banner}")

print("\n\n── FRAUDES (y=1) — scores les plus élevés ──────────────────────────────")
for i, idx in enumerate(top3_fraud, 1):
    print_case(idx, "FRAUDE", i)

print("\n\n── TRANSACTIONS NORMALES (y=0) — scores les plus bas ───────────────────")
for i, idx in enumerate(top3_normal, 1):
    print_case(idx, "NORMALE", i)

print(f"\n\n{banner}")
print("  Rappel : seuil de décision API = 80%")
print("    prob ≥ 0.80  →  REJECTED  (fraude détectée)")
print("    prob < 0.80  →  APPROVED  (transaction normale)")
print(f"{banner}\n")
