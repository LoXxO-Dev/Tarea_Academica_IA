"""Figuras 2 y 3 del informe a partir de results.json.  Uso: python make_figures.py CARPETA_SALIDA"""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

out = sys.argv[1]
for f in glob.glob(os.path.expanduser("~/.fonts/Montserrat-*.ttf")):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Montserrat", "DejaVu Sans"], "font.size": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6,
                     "xtick.color": "#55554f", "ytick.color": "#55554f",
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1c", "#77776f"
R = json.load(open(f"{out}/results.json"))
F = R["final"]
com = lambda v, d=2: f"{v:.{d}f}".replace(".", ",")

# ---------- Fig. 2: recall y precision segun el umbral (fuera de pliegue, M1-S0)
best = R["info"]["best_by_cv_auc"]
cur = F[best]["threshold_curve"]
t = [r[0] for r in cur]; rec = [r[1] for r in cur]
pre = [r[2] if r[1] > 0 else float("nan") for r in cur]   # sin alertas no hay precision
tau = F[best]["tau"]
fig, ax = plt.subplots(figsize=(3.4, 2.15), dpi=400)
ax.plot(t, rec, color=BLUE, lw=1.4)
ax.plot(t, pre, color=ORANGE, lw=1.4, ls=(0, (5, 1.5)))
ax.axhline(0.80, color=MUTED, lw=0.6, ls=(0, (2, 2)))
ax.text(0.06, 0.775, "Objetivo 0,80", ha="left", va="top", color=MUTED, fontsize=5.8)
for x, lab in [(tau, f"τ∗ = {com(tau)}"), (0.5, "τ = 0,50")]:
    ax.axvline(x, color=INK, lw=0.6, ls=(0, (1, 1.5)))
    i = min(range(len(t)), key=lambda k: abs(t[k] - x))
    ax.plot([x], [rec[i]], "o", ms=4, color=BLUE, mec="white", mew=0.8, zorder=5)
    ax.plot([x], [pre[i]], "s", ms=3.8, color=ORANGE, mec="white", mew=0.8, zorder=5)
    ax.text(x + 0.012, 1.02, lab, ha="left", va="bottom", color=INK, fontsize=6, fontweight="bold")
    hi, lo = max(rec[i], pre[i]), min(rec[i], pre[i])
    ax.text(x + 0.015, hi + 0.03, com(hi, 3), color=INK, fontsize=5.8, va="bottom")
    ax.text(x + 0.015, lo - 0.03, com(lo, 3), color=INK, fontsize=5.8, va="top")
ax.text(0.665, 0.36, "Recall", color=INK, fontsize=6.3, fontweight="bold", ha="left", va="center")
ax.text(0.615, 0.645, "Precision", color=INK, fontsize=6.3, fontweight="bold", ha="left", va="center")
ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 1.0)
ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9]); ax.set_xticklabels([com(v, 1) for v in [0.1, 0.3, 0.5, 0.7, 0.9]])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels([com(v, 1) for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]])
ax.set_xlabel("Umbral de decisión τ", color=INK); ax.set_ylabel("Valor de la métrica", color=INK)
ax.grid(axis="y", color="#e6e6e1", lw=0.4); ax.set_axisbelow(True)
fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.95))
fig.savefig(f"{out}/fig2_umbral.png"); plt.close(fig)

# ---------- Fig. 3: curvas ROC en prueba (S0)
fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=400)
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (2, 2)))
styles = [("M2-S0", "M2 Árbol CART", AQUA, (0, (1, 1))), ("M3-S0", "M3 Híbrido", ORANGE, (0, (5, 1.5))),
          ("M1-S0", "M1 Regresión logística", BLUE, "-")]
for key, name, col, ls in styles:
    fpr, tpr = F[key]["roc_test"]
    ax.plot(fpr, tpr, color=col, lw=1.3, ls=ls,
            label=f"{name} (AUC = {com(F[key]['test_at_tau']['auc'], 3)})")
m = F[best]["test_at_tau"]
x0, y0 = m["fp"] / (m["fp"] + m["tn"]), m["recall"]
ax.plot([x0], [y0], "o", ms=4.5, color=BLUE, mec="white", mew=0.9, zorder=6)
ax.annotate(f"M1 con τ∗ = {com(tau)}\nTVP {com(y0, 3)} · TFP {com(x0, 3)}", xy=(x0, y0), xytext=(0.03, 0.90),
            fontsize=5.8, color=INK, arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, shrinkB=3))
ax.text(0.60, 0.545, "Azar (AUC = 0,5)", color=MUTED, fontsize=5.8, rotation=36, ha="center", va="center")
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="lower right", frameon=False, fontsize=5.9, handlelength=2.6, borderaxespad=0.2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
tk = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
ax.set_xticks(tk); ax.set_xticklabels([com(v, 1) for v in tk]); ax.set_yticks(tk); ax.set_yticklabels([com(v, 1) for v in tk])
ax.set_xlabel("Tasa de falsos positivos (TFP)", color=INK); ax.set_ylabel("Tasa de verdaderos positivos (Recall)", color=INK)
ax.grid(color="#e6e6e1", lw=0.4); ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fig.savefig(f"{out}/fig3_roc.png"); plt.close(fig)
print("ok")
