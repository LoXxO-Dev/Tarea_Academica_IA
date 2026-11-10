"""Cifras descriptivas del conjunto de entrenamiento usadas en el informe. Uso: python facts.py CSV CARPETA_SALIDA"""
import json, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from churn_pipeline import clean_and_engineer, make_preprocessor, train_test_split, SEED

csv, out = sys.argv[1], sys.argv[2]
X, y, _ = clean_and_engineer(pd.read_csv(csv))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
d = Xtr.copy(); d["y"] = ytr; d["fiber"] = (d.InternetService == "Fiber optic").astype(int)
seg = json.load(open(f"{out}/results.json"))["final"]["M3-S0"]["segments"]
z = float(re.search(r"tenure <= (-?[\d.]+)", seg).group(1))            # corte en unidades estandarizadas
assert "InternetService_Fiber optic" in seg
cut = int(np.floor(d.tenure.mean() + z * d.tenure.std(ddof=0)))
new = d.tenure <= cut
facts = dict(
    n_feat_drop=len(make_preprocessor(True).fit(Xtr).get_feature_names_out()),
    n_feat_full=len(make_preprocessor(False).fit(Xtr).get_feature_names_out()),
    tenure_sd=float(d.tenure.std(ddof=0)), tenure_cut=cut,
    seg_new_fiber=float(d[new & (d.fiber == 1)].y.mean()), n_new_fiber=int((new & (d.fiber == 1)).sum()),
    seg_old_nofiber=float(d[~new & (d.fiber == 0)].y.mean()), n_old_nofiber=int((~new & (d.fiber == 0)).sum()),
    corr_mc_fiber=float(np.corrcoef(d.MonthlyCharges, d.fiber)[0, 1]),
    corr_mc_avg=float(np.corrcoef(d.MonthlyCharges, d.avg_charge)[0, 1]),
    churn_by_contract=d.groupby("Contract").y.mean().round(4).to_dict(),
)
json.dump(facts, open(f"{out}/facts.json", "w"), indent=1); print(facts)
