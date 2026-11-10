"""
Grupo 4 - 1INF24 - Prediccion de abandono de clientes (churn)
Pipeline completo: (A) limpieza e ingenieria de atributos, (B) particion estratificada,
(C) codificacion, escalamiento y desbalance, (D) modelos M1-M3, (E) ajuste del umbral.

Uso:  python churn_pipeline.py RUTA_AL_CSV CARPETA_SALIDA
Requiere: numpy, pandas, scikit-learn.
"""
import json
import sys
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42

SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
BINARY = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
          "MultipleLines", "PaperlessBilling"] + SERVICES
NOMINAL = ["InternetService", "Contract", "PaymentMethod"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "n_services",
           "avg_charge", "charge_diff"]
FEATURES = BINARY + NOMINAL + NUMERIC


# ---------------------------------------------------------------- (A)
def clean_and_engineer(raw):
    """Reglas fijas fila por fila; no se calcula ninguna estadistica del conjunto."""
    df = raw.copy()
    total = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    n_blank = int(total.isna().sum())
    assert (df.loc[total.isna(), "tenure"] == 0).all(), "vacios con tenure != 0"
    df["TotalCharges"] = total.fillna(0.0)
    df = df.replace({"No internet service": "No", "No phone service": "No"})
    df["n_services"] = (df[SERVICES] == "Yes").sum(axis=1).astype(float)
    df["avg_charge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["charge_diff"] = df["MonthlyCharges"] - df["avg_charge"]
    for c in BINARY:
        if c == "gender":
            df[c] = (df[c] == "Male").astype(int)
        elif c != "SeniorCitizen":
            df[c] = (df[c] == "Yes").astype(int)
    df["tenure"] = df["tenure"].astype(float)
    y = (df["Churn"] == "Yes").astype(int).to_numpy()
    return df[FEATURES].reset_index(drop=True), y, n_blank


# ---------------------------------------------------------------- (C) SMOTE-NC
def smote_nc(X, y, k=5, seed=SEED):
    """SMOTE para datos mixtos (Chawla et al., 2002, variante NC).
    Distancia: euclidiana sobre numericas estandarizadas (estadisticas del propio
    conjunto de entrenamiento recibido) + med^2 por cada categorica distinta, donde
    med es la mediana de las desviaciones de las numericas en la clase minoritaria.
    Numericas: interpolacion; categoricas: moda de los k vecinos.
    n_services se recalcula a partir de los servicios sinteticos."""
    rng = np.random.default_rng(seed)
    cat_cols = BINARY + NOMINAL
    num_cols = [c for c in NUMERIC if c != "n_services"]
    Xmin = X[y == 1].reset_index(drop=True)
    n_new = int((y == 0).sum() - (y == 1).sum())
    if n_new <= 0:
        return X, y
    mu, sd = X[num_cols].mean(), X[num_cols].std(ddof=0).replace(0, 1.0)
    Z = ((Xmin[num_cols] - mu) / sd).to_numpy()
    med = float(np.median(Z.std(axis=0)))
    C = pd.get_dummies(Xmin[cat_cols].astype(str)).to_numpy(dtype=float) * med / np.sqrt(2)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(np.hstack([Z, C]))
    neigh = nn.kneighbors(np.hstack([Z, C]), return_distance=False)[:, 1:]
    base = rng.integers(0, len(Xmin), n_new)
    pick = neigh[base, rng.integers(0, k, n_new)]
    gap = rng.random(n_new)[:, None]
    num_base = Xmin[num_cols].to_numpy()
    new_num = num_base[base] + gap * (num_base[pick] - num_base[base])
    syn = pd.DataFrame(new_num, columns=num_cols)
    for c in cat_cols:
        vals = Xmin[c].to_numpy()[neigh[base]]            # (n_new, k)
        syn[c] = pd.DataFrame(vals).mode(axis=1)[0].to_numpy()
        syn[c] = syn[c].astype(Xmin[c].dtype)
    syn["n_services"] = syn[SERVICES].sum(axis=1).astype(float)
    Xr = pd.concat([X, syn[X.columns]], ignore_index=True)
    yr = np.concatenate([y, np.ones(n_new, dtype=int)])
    return Xr, yr


# ---------------------------------------------------------------- (C) codificacion + escalamiento
def make_preprocessor(drop_first):
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("nom", OneHotEncoder(drop="first" if drop_first else None,
                              handle_unknown="ignore", sparse_output=False), NOMINAL),
        ("bin", "passthrough", BINARY),
    ], verbose_feature_names_out=False)


# ---------------------------------------------------------------- (D) modelos
class LogitLeaf:
    """M3: arbol poco profundo que segmenta + regresion logistica por hoja."""

    def __init__(self, depth, C, class_weight):
        self.depth, self.C, self.class_weight = depth, C, class_weight

    def fit(self, X, y):
        self.tree = DecisionTreeClassifier(max_depth=self.depth, min_samples_leaf=0.10,
                                           class_weight=self.class_weight,
                                           random_state=SEED).fit(X, y)
        leaves = self.tree.apply(X)
        self.models = {}
        for leaf in np.unique(leaves):
            m = leaves == leaf
            if len(np.unique(y[m])) < 2:
                self.models[leaf] = float(y[m].mean())
            else:
                self.models[leaf] = LogisticRegression(
                    C=self.C, penalty="l2", class_weight=self.class_weight,
                    max_iter=2000).fit(X[m], y[m])
        return self

    def predict_proba(self, X):
        leaves = self.tree.apply(X)
        p = np.zeros(len(X))
        for leaf, mdl in self.models.items():
            m = leaves == leaf
            if m.any():
                p[m] = mdl if isinstance(mdl, float) else mdl.predict_proba(X[m])[:, 1]
        return np.column_stack([1 - p, p])


GRIDS = {
    "M1": [dict(C=C, penalty=p) for C, p in product([0.01, 0.1, 1, 10], ["l1", "l2"])],
    "M2": [dict(max_depth=d, min_samples_leaf=m, ccp_alpha=a)
           for d, m, a in product([3, 4, 5, 6, 8, 10], [5, 10, 20, 50],
                                  [0, 0.001, 0.005, 0.01])],
    "M3": [dict(depth=d, C=C) for d, C in product([2, 3], [0.1, 1, 10])],
}


def build(model, params, class_weight):
    if model == "M1":
        return LogisticRegression(C=params["C"], penalty=params["penalty"],
                                  solver="liblinear", class_weight=class_weight,
                                  max_iter=2000, random_state=SEED)
    if model == "M2":
        return DecisionTreeClassifier(**params, class_weight=class_weight,
                                      random_state=SEED)
    return LogitLeaf(params["depth"], params["C"], class_weight)


def prepare(Xtr, ytr, strategy, drop_first, seed):
    """Todo lo que aprende de los datos se ajusta solo con (Xtr, ytr)."""
    if strategy == "S2":
        Xtr, ytr = smote_nc(Xtr, ytr, seed=seed)
    pre = make_preprocessor(drop_first).fit(Xtr)
    return pre, pre.transform(Xtr), ytr


def choose_threshold(y, p, target=0.80):
    """Ec. (3): max. precision s.a. recall >= target; si no hay, max. F1."""
    best, rule = None, "recall>=0.80"
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        rows.append((t, recall_score(y, pred), precision_score(y, pred, zero_division=0),
                     f1_score(y, pred, zero_division=0)))
    ok = [r for r in rows if r[1] >= target]
    if ok:
        best = max(ok, key=lambda r: (r[2], r[0]))
    else:
        best, rule = max(rows, key=lambda r: r[3]), "max F1"
    return float(best[0]), rule, rows


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(recall=recall_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), auc=roc_auc_score(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def main(csv_path, out_dir):
    t_start = time.time()
    raw = pd.read_csv(csv_path)
    X, y, n_blank = clean_and_engineer(raw)
    info = dict(n=len(y), churners=int(y.sum()), pi=float(y.mean()), blank_total=n_blank)

    # (B) particion estratificada 80/20 y pliegues
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))
    info.update(n_train=len(ytr), n_test=len(yte), churn_train=int(ytr.sum()),
                churn_test=int(yte.sum()))

    cv_rows, oof, final = [], {}, {}
    for model, strategy in product(["M1", "M2", "M3"], ["S0", "S1", "S2"]):
        t0 = time.time()
        drop_first = model != "M2"
        # S1: pesos globales w_c = N / (2 N_c), calculados con el entrenamiento
        cw = ({c: len(ytr) / (2 * (ytr == c).sum()) for c in (0, 1)}
              if strategy == "S1" else None)
        grid = GRIDS[model]
        aucs = np.zeros((len(grid), 5))
        preds = np.zeros((len(grid), len(ytr)))
        for f, (itr, iva) in enumerate(folds):
            pre, A, b = prepare(Xtr.iloc[itr].reset_index(drop=True), ytr[itr],
                                strategy, drop_first, SEED + f)
            V = pre.transform(Xtr.iloc[iva])
            for g, params in enumerate(grid):
                p = build(model, params, cw).fit(A, b).predict_proba(V)[:, 1]
                preds[g, iva] = p
                aucs[g, f] = roc_auc_score(ytr[iva], p)
        for g, params in enumerate(grid):
            cv_rows.append(dict(model=model, strategy=strategy, **params,
                                auc_mean=aucs[g].mean(), auc_std=aucs[g].std()))
        g_best = int(np.argmax(aucs.mean(axis=1)))
        params = grid[g_best]
        p_oof = preds[g_best]
        tau, rule, curve = choose_threshold(ytr, p_oof)
        fold_recall = lambda t: [float(recall_score(ytr[iva], (p_oof[iva] >= t).astype(int)))
                                 for _, iva in folds]
        fold_prec = lambda t: [float(precision_score(ytr[iva], (p_oof[iva] >= t).astype(int),
                                                     zero_division=0)) for _, iva in folds]
        # modelo final: reentrenamiento con el 80 % y evaluacion unica en prueba
        pre, A, b = prepare(Xtr, ytr, strategy, drop_first, SEED)
        clf = build(model, params, cw).fit(A, b)
        p_te = clf.predict_proba(pre.transform(Xte))[:, 1]
        key = f"{model}-{strategy}"
        oof[key] = p_oof
        final[key] = dict(
            model=model, strategy=strategy, params=params,
            cv_auc_mean=float(aucs[g_best].mean()), cv_auc_std=float(aucs[g_best].std()),
            cv_auc_folds=[float(a) for a in aucs[g_best]],
            tau=tau, tau_rule=rule,
            oof_at_tau=metrics(ytr, p_oof, tau), oof_at_05=metrics(ytr, p_oof, 0.5),
            fold_recall_tau=fold_recall(tau), fold_recall_05=fold_recall(0.5),
            fold_prec_tau=fold_prec(tau), fold_prec_05=fold_prec(0.5),
            test_at_tau=metrics(yte, p_te, tau), test_at_05=metrics(yte, p_te, 0.5),
            seconds=round(time.time() - t0, 1),
            threshold_curve=[[float(v) for v in r] for r in curve],
            roc_test=[a.tolist() for a in roc_curve(yte, p_te)[:2]],
            p_test=p_te.tolist(),
        )
        names = list(pre.get_feature_names_out())
        if model == "M1":
            coef = clf.coef_[0]
            final[key]["coef"] = sorted(
                [dict(feature=n, w=float(w), odds=float(np.exp(w))) for n, w in zip(names, coef)],
                key=lambda d: -abs(d["w"]))
            final[key]["n_zero_coef"] = int((coef == 0).sum())
        if model == "M2":
            final[key]["importances"] = sorted(
                [dict(feature=n, imp=float(i)) for n, i in zip(names, clf.feature_importances_) if i > 0],
                key=lambda d: -d["imp"])
            final[key]["n_leaves"] = int(clf.get_n_leaves())
            final[key]["rules"] = export_text(clf, feature_names=names, max_depth=3)
        if model == "M3":
            final[key]["segments"] = export_text(clf.tree, feature_names=names)
            final[key]["n_segments"] = int(clf.tree.get_n_leaves())
        print(f"{key}: AUC cv {aucs[g_best].mean():.4f} | tau {tau:.2f} ({rule}) | "
              f"test R {final[key]['test_at_tau']['recall']:.3f} "
              f"P {final[key]['test_at_tau']['precision']:.3f} "
              f"AUC {final[key]['test_at_tau']['auc']:.4f} | {final[key]['seconds']} s", flush=True)

    # referencia trivial: siempre la clase mayoritaria
    info["trivial"] = dict(accuracy=float((yte == 0).mean()), recall=0.0, precision=0.0,
                           f1=0.0, auc=0.5)
    # equidad: desempeno del mejor modelo por grupo (se elige por AUC de validacion)
    best_key = max(final, key=lambda k: final[k]["cv_auc_mean"])
    info["best_by_cv_auc"] = best_key
    pb, tb = np.array(final[best_key]["p_test"]), final[best_key]["tau"]
    fair = {}
    for col, labels in [("gender", {0: "Mujer", 1: "Hombre"}),
                        ("SeniorCitizen", {0: "No adulto mayor", 1: "Adulto mayor"})]:
        for v, lab in labels.items():
            m = (Xte[col] == v).to_numpy()
            pred = (pb[m] >= tb).astype(int)
            fair[lab] = dict(n=int(m.sum()), churn_rate=float(yte[m].mean()),
                             recall=float(recall_score(yte[m], pred)),
                             precision=float(precision_score(yte[m], pred, zero_division=0)),
                             flagged=float(pred.mean()))
    info["fairness"] = fair
    # variante parsimoniosa de M1 (L1, C = 0.1) para la lectura de coeficientes
    pre, A, b_ = prepare(Xtr, ytr, "S0", True, SEED)
    l1 = build("M1", dict(C=0.1, penalty="l1"), None).fit(A, b_)
    info["m1_l1"] = dict(
        n_features=int(len(l1.coef_[0])), n_zero=int((l1.coef_[0] == 0).sum()),
        test_auc=float(roc_auc_score(yte, l1.predict_proba(pre.transform(Xte))[:, 1])),
        coef=sorted([dict(feature=n, w=float(w), odds=float(np.exp(w)))
                     for n, w in zip(pre.get_feature_names_out(), l1.coef_[0]) if w != 0],
                    key=lambda d: -abs(d["w"])))
    info["total_seconds"] = round(time.time() - t_start, 1)

    pd.DataFrame(cv_rows).to_csv(f"{out_dir}/cv_grid.csv", index=False)
    with open(f"{out_dir}/results.json", "w") as fh:
        json.dump(dict(info=info, final=final), fh, indent=1)
    print("Mejor por AUC de validacion:", best_key, "| tiempo total:", info["total_seconds"], "s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
