"""
Redraws manuscript Figures 1-6 from the outputs of full_pipeline.py.
Usage: python make_figures.py punjab_cgwb_dtw_wellwise_parsed.csv results_folder figures_folder
Numbering follows the manuscript: Fig 1 coverage/distribution, Fig 2 district change, Fig 3 gauged thinning,
Fig 4 ungauged thinning, Fig 5 (a) value of history (b) district RMSE, Fig 6 interval calibration.
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV, RES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
                     "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
C = {"pers": "#7f7f7f", "sn": "#8c564b", "sn_dist": "#bcbd22", "shift": "#1f77b4", "ridge": "#2ca02c", "rf": "#9467bd", "gb": "#d62728",
     "global": "#7f7f7f", "district": "#1f77b4", "shrunk": "#2ca02c", "ens": "#d62728", "conf": "#1f4e9c"}
NAMES = {"pers": "Persistence", "sn": "Seasonal-naive", "sn_dist": "Seasonal-naive + district shift", "shift": "District-shift",
         "ridge": "Ridge", "rf": "Random forest", "gb": "Boosting", "global": "Global mean", "district": "District mean",
         "shrunk": "Shrunk district mean"}
def rd(n, **k): return pd.read_csv(os.path.join(RES, n + ".csv"), **k)
def nice(s): return {"SAS NAGAR": "SAS Nagar", "SBS NAGAR": "SBS Nagar", "TARAN TARAN": "Taran Taran"}.get(s, s.title())
def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight"); plt.close(fig); print("saved", name)

# ------------------------------------------------------------------ Fig 1
raw = pd.read_csv(CSV); R = [c for c in raw.columns if c[:2] == "20"]
raw["range"] = raw[R].max(axis=1) - raw[R].min(axis=1); ok = raw[raw.range <= 12]
LAB = {"2019-06": "Jun 19", "2019-08": "Aug 19", "2019-10": "Oct 19", "2020-01": "Jan 20", "2022-06": "Jun 22", "2022-08": "Aug 22",
       "2022-11": "Nov 22", "2023-01": "Jan 23", "2023-06": "Jun 23", "2023-08": "Aug 23", "2023-11": "Nov 23", "2024-01": "Jan 24"}
cols3 = ["#4c78a8"] * 4 + ["#f58518"] * 4 + ["#54a24b"] * 4
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax[0].bar(range(12), [ok[c].notna().sum() for c in R], color=cols3)
ax[0].set_xticks(range(12)); ax[0].set_xticklabels([LAB[c] for c in R], rotation=60, ha="right")
ax[0].set_ylabel("Wells with a reading"); ax[0].set_title("(a) Network coverage per round")
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color="#4c78a8", label="2019–20"), Patch(color="#f58518", label="2022–23"), Patch(color="#54a24b", label="2023–24")], frameon=False, loc="upper right")
ax[1].boxplot([ok[c].dropna() for c in R], flierprops=dict(marker="o", markersize=2), medianprops=dict(color="k"))
ax[1].set_xticklabels([LAB[c] for c in R], rotation=60, ha="right"); ax[1].set_ylabel("Depth to water level (m bgl)")
ax[1].set_title("(b) Distribution of readings"); fig.tight_layout(); save(fig, "Fig1_coverage_distribution")

# ------------------------------------------------------------------ Fig 2 (map + district bars)
# Approximate district headquarters (lon, lat). DRAFT VALUES: verify against an official source (e.g. Census of India / Survey of India)
# before publication, or replace this bubble map with a boundary map (e.g. GADM level-2 India, Punjab districts).
HQ = {"AMRITSAR": (74.87, 31.63), "BARNALA": (75.55, 30.38), "BATHINDA": (74.95, 30.21), "FARIDKOT": (74.76, 30.67),
      "FATEHGARH SAHIB": (76.39, 30.64), "FAZILKA": (74.03, 30.40), "FIROZPUR": (74.61, 30.93), "GURDASPUR": (75.40, 32.04),
      "HOSHIARPUR": (75.91, 31.53), "JALANDHAR": (75.58, 31.33), "KAPURTHALA": (75.38, 31.38), "LUDHIANA": (75.85, 30.90),
      "MANSA": (75.39, 29.99), "MOGA": (75.17, 30.82), "MUKTSAR": (74.52, 30.47), "PATHANKOT": (75.65, 32.27),
      "PATIALA": (76.39, 30.34), "RUPNAGAR": (76.53, 30.97), "SANGRUR": (75.84, 30.25), "SAS NAGAR": (76.72, 30.70),
      "SBS NAGAR": (76.12, 31.12), "TARAN TARAN": (74.93, 31.45)}
t3all = rd("table3_district_change").set_index("district"); nwell = ok.groupby("district")["2023-11"].count()
fig, ax = plt.subplots(1, 2, figsize=(7.6, 4.2), gridspec_kw={"width_ratios": [1.05, 1]})
a = ax[0]; lat0 = 31.1
for dname, (lo, la) in HQ.items():
    n = nwell.get(dname, 0)
    if dname in t3all.index:
        v = t3all.loc[dname, "median"]; a.scatter(lo, la, s=n * 3.2, c=[v], cmap="RdBu_r", vmin=-2.5, vmax=2.5, edgecolor="k", lw=0.5, zorder=3)
    else:
        a.scatter(lo, la, s=n * 3.2, facecolor="none", edgecolor="#555", lw=0.8, ls="--", zorder=3)
    OFF = {"KAPURTHALA": (0, 8, "center"), "JALANDHAR": (0, -14, "center"), "FATEHGARH SAHIB": (6, -9, "left"), "SBS NAGAR": (6, -3, "left"),
           "TARAN TARAN": (-6, 0, "right"), "BARNALA": (-3, 6, "right")}
    dx, dy, ha = OFF.get(dname, (4, 4, "left"))
    a.annotate(nice(dname), (lo, la), xytext=(dx, dy), textcoords="offset points", fontsize=5.5, ha=ha)
a.set_xlim(73.8, 77.3); a.set_ylim(29.8, 32.45); a.set_aspect(1 / np.cos(np.radians(lat0))); a.set_xlabel("Longitude (°E)"); a.set_ylabel("Latitude (°N)")
a.set_title("(a) Wells with a Nov 2023 reading (size)\nand median June change 2019–23 (colour)", fontsize=8)
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-2.5, 2.5)); cb = fig.colorbar(sm, ax=a, orientation="horizontal", fraction=0.05, pad=0.16, shrink=0.8); cb.set_label("Median change in June depth, 2019–23 (m); dashed = fewer than 8 matched wells", fontsize=6); cb.ax.tick_params(labelsize=6)
for n_ in (10, 30, 50): a.scatter([], [], s=n_ * 3.2, facecolor="none", edgecolor="k", lw=0.5, label=f"{n_} wells")
a.legend(frameon=False, loc="upper left", fontsize=6, scatterpoints=1, labelspacing=1.3, borderpad=0.2)
t3 = t3all.reset_index().sort_values("median")
ax[1].barh([f"{nice(d)} ({int(n)})" for d, n in zip(t3.district, t3.n)], t3["median"], color=np.where(t3["median"] > 0, "#c0392b", "#2e6da4"))
ax[1].axvline(0, color="k", lw=0.6); ax[1].set_xlabel("Median change in June depth to water level,\n2019 to 2023 (m)"); ax[1].set_title("(b) Districts with at least 8 matched wells", fontsize=8)
ax[1].tick_params(axis="y", labelsize=6.5); fig.tight_layout(); save(fig, "Fig2_district_map_and_change")

# ------------------------------------------------------------------ Fig 3 / Fig 4 (thinning with bands)
def thin_panel(ax, df, models, band_models, title):
    g = df.groupby(["f", "model"]).rmse.agg(mean="mean", lo=lambda s: s.quantile(.025), hi=lambda s: s.quantile(.975)).reset_index()
    for m in models:
        d = g[g.model == m].sort_values("f", ascending=False); x = d.f * 100
        flat = m in ("pers", "sn", "global")
        ax.plot(x, d["mean"], marker="o", ms=3, lw=1.4, ls="--" if flat else "-", color=C[m], label=NAMES[m])
        if m in band_models: ax.fill_between(x, d.lo, d.hi, color=C[m], alpha=0.12, lw=0)
    ax.set_xlim(104, 6); ax.set_xticks([100, 75, 50, 25, 10]); ax.set_xlabel("Training wells retained (%)"); ax.set_ylabel("RMSE on withheld wells (m)"); ax.set_title(title)
for fig_name, prefix, models, bands, tt, ylim in [
        ("Fig3_gauged_thinning", "splits_scenarioB_", ["pers", "sn", "shift", "ridge", "rf", "gb"], ["ridge", "gb"], ("(a) Target: Nov 2023", "(b) Target: Jun 2023"), None),
        ("Fig4_ungauged_thinning", "splits_scenarioA_", ["global", "district", "shrunk", "gb"], ["district", "gb"], ("(a) Target: Nov 2023", "(b) Target: Jun 2023"), None)]:
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=False)
    for a, tgt, t in zip(ax, ["2023-11", "2023-06"], tt):
        thin_panel(a, pd.read_csv(os.path.join(RES, f"{prefix}{tgt}.csv")), models, bands, t)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=len(models), loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, fig_name)

# ------------------------------------------------------------------ Fig 5 (history + district RMSE)
t7 = rd("table7_value_of_history"); fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
labs = ["Jun-23\nonly", "+ Nov-22", "+ Jun-22", "+ 2019\nrounds", "Prior years\nonly*"]; x = np.arange(len(t7)); w = 0.2
for i, (c, col, nm) in enumerate([("Persistence", C["pers"], "Persistence"), ("District-shift", C["shift"], "District-shift"), ("Ridge", C["ridge"], "Ridge"), ("Boosting", C["gb"], "Boosting")]):
    ax[0].bar(x + (i - 1.5) * w, t7[c], w, color=col, label=nm)
ax[0].set_xticks(x); ax[0].set_xticklabels(labs, fontsize=7); ax[0].set_ylabel("RMSE, Nov 2023 (m)"); ax[0].set_ylim(1.0, 2.75)
ax[0].set_title("(a) Value of a well's own history"); ax[0].legend(frameon=False, ncol=2, loc="upper right"); ax[0].text(0.0, -0.22, "*different set of wells", transform=ax[0].transAxes, fontsize=6.5)
dd = rd("fig_data_district_rmse_ungauged"); dd = dd[dd.target == "2023-11"].pivot(index="district", columns="f", values="rmse").sort_values(1.0)
y = np.arange(len(dd)); ax[1].barh(y + 0.2, dd[1.0], 0.38, color="#1f77b4", label="100% retained"); ax[1].barh(y - 0.2, dd[0.25], 0.38, color="#f39c12", label="25% retained")
ax[1].set_yticks(y); ax[1].set_yticklabels([nice(s) for s in dd.index], fontsize=6.5); ax[1].set_xlabel("District-mean RMSE, Nov 2023 (m)")
ax[1].set_title("(b) Ungauged wells by district"); ax[1].legend(frameon=False, loc="lower right"); fig.tight_layout(); save(fig, "Fig5_history_and_district_rmse")

# ------------------------------------------------------------------ Fig 6 (calibration, four interval methods)
U = pd.read_csv(os.path.join(RES, "raw_uncertainty.csv")); cov = U[U.level != "crps"].copy(); cov["level"] = cov.level.astype(float)
MC = {"ens": ("#d62728", "Bootstrap ensemble"), "gcv": ("#2ca02c", "Gaussian (CV residual SD)"), "qgb": ("#ff7f0e", "Quantile boosting"), "conf": ("#1f4e9c", "Conformal (CV residuals)")}
fig, ax = plt.subplots(1, 3, figsize=(7.8, 3.0))
for tgt, ls in [("2023-11", "-"), ("2023-06", "--")]:
    s9 = cov[(cov.target == tgt) & (cov.level == 0.9)].groupby("f")[[f"{p}_{k}" for p in ("picp", "width") for k in MC]].mean().sort_index(ascending=False)
    for k, (col, lab) in MC.items():
        ax[0].plot(s9.index * 100, s9[f"picp_{k}"], ls, color=col, marker="o", ms=2.5); ax[2].plot(s9.index * 100, s9[f"width_{k}"], ls, color=col, marker="o", ms=2.5)
ax[0].axhline(0.9, color="k", ls=":", lw=0.8); ax[0].set_ylim(0.3, 1.0); ax[0].set_ylabel("Coverage of nominal 90% interval"); ax[0].set_title("(a) Coverage")
r = cov[(cov.target == "2023-11") & (cov.f == 0.25)].groupby("level")[["picp_ens", "picp_gcv", "picp_conf"]].mean()
ax[1].plot([0.4, 1], [0.4, 1], "k:", lw=0.8)
for k in ("ens", "gcv", "conf"): ax[1].plot(r.index, r[f"picp_{k}"], color=MC[k][0], marker="o", ms=2.5)
q9 = cov[(cov.target == "2023-11") & (cov.f == 0.25) & (cov.level == 0.9)].picp_qgb.mean(); ax[1].plot([0.9], [q9], marker="o", ms=4, color=MC["qgb"][0], ls="")
ax[1].set_xlabel("Nominal coverage"); ax[1].set_ylabel("Empirical coverage"); ax[1].set_title("(b) Reliability, 25% retained"); ax[1].set_xlim(0.45, 1); ax[1].set_ylim(0.1, 1)
ax[2].set_ylabel("Mean 90% interval width (m)"); ax[2].set_title("(c) Width")
for a_ in (ax[0], ax[2]): a_.set_xlim(104, 6); a_.set_xticks([100, 50, 25, 10]); a_.set_xlabel("Training wells retained (%)")
from matplotlib.lines import Line2D
h = [Line2D([0], [0], color=c_, marker="o", ms=3, label=l_) for c_, l_ in MC.values()] + [Line2D([0], [0], color="k", ls="-", label="Nov 2023"), Line2D([0], [0], color="k", ls="--", label="Jun 2023")]
fig.legend(handles=h, ncol=6, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=6.5, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "Fig6_interval_calibration")
