"""
Full reproducible pipeline: Punjab CGWB year-book depth-to-water data.
Regenerates every table in the manuscript (Tables 1-8, 4b, A1, A2) from the parsed CSV.

Usage:
    python full_pipeline.py punjab_cgwb_dtw_wellwise_parsed.csv  results_folder  [n_splits] [n_uq_reps]
    (defaults: 40 random splits for Tables 4-7 and A1-A2, 25 repetitions for Table 8)

Quality control (explicit, applied everywhere):
    - records with a within-edition range above 12 m are excluded (RANGE_MAX)
    - records whose normalised name is duplicated within a district and edition are excluded
      from the cross-year panel used for modelling and matching (they stay in Table 1 counts)
Random numbers: one master seed (SEED) so every run gives identical tables.
"""
import sys, os, warnings, time
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, norm
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
warnings.filterwarnings("ignore")

CSV = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
N_SPLITS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
N_UQ = int(sys.argv[4]) if len(sys.argv) > 4 else 25
SEED, RANGE_MAX, TEST_FRAC = 2024, 12.0, 0.25
RETAIN = [1.0, 0.75, 0.5, 0.25, 0.10]
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
def save(df, name, idx=True):
    df.to_csv(os.path.join(OUT, name + ".csv"), index=idx); log(f"saved {name}.csv")

# ------------------------------------------------------------------ data + QC
raw = pd.read_csv(CSV)
ROUNDS = [c for c in raw.columns if c[:2] == "20"]
raw["key"] = raw.district + "|" + raw.well.str.lower().str.replace("[^a-z0-9]", "", regex=True)
raw["range"] = raw[ROUNDS].max(axis=1) - raw[ROUNDS].min(axis=1)
raw["dup"] = raw.duplicated(["file", "key"], keep=False)
ok = raw[raw.range <= RANGE_MAX]                                  # for Table 1
pan = raw[(raw.range <= RANGE_MAX) & ~raw.dup]                    # panel for matching / modelling
log(f"records {len(raw)}; range>{RANGE_MAX:.0f} m excluded: {(raw.range > RANGE_MAX).sum()}; duplicate-name records: {raw.dup.sum()}")
d = pan.groupby("key")[ROUNDS].first()
d["district"] = pan.groupby("key").district.first()
nm = d.index.str.split("|").str[1]
d["type"] = np.where(nm.str.contains("ml"), "ml", np.where(nm.str.contains("pz"), "pz", "dw"))
LAB = {"2019-06": "Jun 2019", "2019-08": "Aug 2019", "2019-10": "Oct 2019", "2020-01": "Jan 2020",
       "2022-06": "Jun 2022", "2022-08": "Aug 2022", "2022-11": "Nov 2022", "2023-01": "Jan 2023",
       "2023-06": "Jun 2023", "2023-08": "Aug 2023", "2023-11": "Nov 2023", "2024-01": "Jan 2024"}
log(f"panel: {len(d)} unique wells; in all three editions: {(pan.groupby('key').file.nunique() == 3).sum()}")

# ------------------------------------------------------------------ Tables 1-3
t1 = pd.DataFrame({LAB[c]: {"Wells": ok[c].notna().sum(), "Median": ok[c].median(),
        "Q1": ok[c].quantile(.25), "Q3": ok[c].quantile(.75), "Max": ok[c].max()} for c in ROUNDS}).T.round(1)
t1["Wells"] = t1["Wells"].astype(int); save(t1, "table1_coverage")

rows = []
for a, b in [("2019-06", "2022-06"), ("2022-06", "2023-06"), ("2019-06", "2023-06"),
             ("2019-10", "2022-11"), ("2022-11", "2023-11"), ("2019-10", "2023-11")]:
    m = d[[a, b]].dropna(); x = m[b] - m[a]
    rows.append([f"{LAB[a]} -> {LAB[b]}", len(m), x.median(), x.quantile(.25), x.quantile(.75), 100 * (x > 0).mean(), wilcoxon(x).pvalue])
save(pd.DataFrame(rows, columns=["Comparison", "n", "Median", "Q1", "Q3", "% deeper", "p"]).round(3), "table2_paired_change", False)

m = d[["2019-06", "2023-06", "district"]].dropna(); m["c"] = m["2023-06"] - m["2019-06"]
t3 = m.groupby("district").c.agg(n="count", median="median", mean="mean", pct_deeper=lambda s: 100 * (s > 0).mean())
save(t3[t3.n >= 8].sort_values("median", ascending=False).round(2), "table3_district_change")

# Table 9: depth-threshold exposure at matched wells (June 2019 vs June 2023)
mm = d[["2019-06", "2023-06", "district"]].dropna()
save(pd.DataFrame([["Punjab (all matched wells)", f"> {t} m", len(mm), 100 * (mm["2019-06"] > t).mean(), 100 * (mm["2023-06"] > t).mean()]
                   for t in (20, 30, 40)], columns=["Group", "Threshold", "n", "Jun 2019 (% of wells)", "Jun 2023 (% of wells)"]).round(1),
     "table9a_exposure_overall", False)
t9 = mm.groupby("district").apply(lambda s: pd.Series({"n": len(s), "Jun 2019 > 30 m (%)": 100 * (s["2019-06"] > 30).mean(),
        "Jun 2023 > 30 m (%)": 100 * (s["2023-06"] > 30).mean(), "Jun 2023 > 40 m (%)": 100 * (s["2023-06"] > 40).mean()}))
t9["Change > 30 m (percentage points)"] = t9["Jun 2023 > 30 m (%)"] - t9["Jun 2019 > 30 m (%)"]
save(t9[t9.n >= 8].sort_values("Jun 2023 > 30 m (%)", ascending=False).round(0), "table9b_exposure_by_district")

# ------------------------------------------------------------------ helpers
def split(n, rs):
    idx = rs.permutation(n); nt = int(round(TEST_FRAC * n)); return idx[:nt], idx[nt:]
def thin(pool_idx, f, rs):
    k = len(pool_idx) if f >= 1 else max(5, int(round(f * len(pool_idx))))
    return pool_idx[rs.permutation(len(pool_idx))[:k]]
def rmse(e): return float(np.sqrt(np.mean(e ** 2)))

def panel_B(target, anchor_cols, feat_hist):
    t = d[d[target].notna()].copy()
    t["anchor"] = t[anchor_cols].bfill(axis=1).iloc[:, 0]
    t = t[t.anchor.notna()]
    t["y"] = t[target] - t.anchor
    for h in feat_hist: t["d_" + h] = t[h] - t.anchor
    return t, ["anchor"] + ["d_" + h for h in feat_hist]

def shrunk_district_shift(base_df, y, te, k):
    g = y.mean(); grp = y.groupby(base_df.district).agg(["sum", "count"])
    return te.district.map(lambda x: (grp.loc[x, "sum"] + k * g) / (grp.loc[x, "count"] + k) if x in grp.index else g).values

def design(tr, feats):
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(tr[["district", "type"]])
    sc = StandardScaler().fit(tr[feats].fillna(0))
    return (lambda a: np.hstack([sc.transform(a[feats].fillna(0)), oh.transform(a[["district", "type"]])]),
            lambda a: np.hstack([a[feats].fillna(0).values, oh.transform(a[["district", "type"]])]),
            lambda a: np.hstack([a[feats].values, oh.transform(a[["district", "type"]])]))

def gb(seed): return HistGradientBoostingRegressor(max_iter=80, learning_rate=0.06, max_depth=3, min_samples_leaf=8,
                                                   l2_regularization=1.0, random_state=seed)

def predict_B(tr, te, feats, target, seasonal, which, seed):
    p = {"pers": te.anchor.values}
    if "shift" in which: p["shift"] = te.anchor.values + shrunk_district_shift(tr, tr.y, te, 5)
    if seasonal:
        has = te[seasonal].notna().values
        sn = np.where(has, te[seasonal].values, te.anchor.values); p["sn"] = sn
        trs = tr[tr[seasonal].notna()]
        p["sn_dist"] = sn + np.where(has, shrunk_district_shift(trs, trs[target] - trs[seasonal], te, 5), 0)
    Zr, Zt, Zg = design(tr, feats)
    if "ridge" in which: p["ridge"] = te.anchor.values + Ridge(alpha=10).fit(Zr(tr), tr.y).predict(Zr(te))
    if "rf" in which:
        p["rf"] = te.anchor.values + RandomForestRegressor(150, min_samples_leaf=5, random_state=seed, n_jobs=-1).fit(Zt(tr), tr.y).predict(Zt(te))
    if "gb" in which: p["gb"] = te.anchor.values + gb(seed).fit(Zg(tr), tr.y).predict(Zg(te))
    return p

HIST = {"2023-11": ["2023-06", "2022-11", "2022-06", "2019-10", "2019-06"],
        "2023-06": ["2022-11", "2022-06", "2019-10", "2019-06"]}
SEAS = {"2023-11": "2022-11", "2023-06": "2022-06"}
ALL_B = ["shift", "ridge", "rf", "gb"]
NAMES = {"pers": "Persistence", "sn": "Seasonal-naive", "sn_dist": "Seasonal-naive + district shift", "shift": "District-shift",
         "ridge": "Ridge regression", "rf": "Random forest", "gb": "Gradient boosting"}

# ------------------------------------------------------------------ Scenario B: Tables 4, 4b, 5, A1
resB, resB_sub, split_wins = {}, {}, {}
for tgt in ["2023-11", "2023-06"]:
    t, feats = panel_B(tgt, HIST[tgt], HIST[tgt]); seas = SEAS[tgt]
    log(f"Scenario B {tgt}: {len(t)} wells with history")
    rs = np.random.RandomState(SEED + (0 if tgt == "2023-11" else 1))
    rec = []
    for s in range(N_SPLITS):
        te_i, pool_i = split(len(t), rs); te = t.iloc[te_i]; yobs = te[tgt].values
        has = te[seas].notna().values
        for f in RETAIN:
            tr = t.iloc[thin(pool_i, f, rs)]
            p = predict_B(tr, te, feats, tgt, seas, ALL_B, SEED + s)
            for k, v in p.items():
                e = v - yobs
                rec.append(dict(split=s, f=f, model=k, mae=np.abs(e).mean(), rmse=rmse(e), bias=e.mean(),
                                rmse_sub=rmse(e[has]), mae_sub=np.abs(e[has]).mean(), n_test=len(te), n_sub=int(has.sum()),
                                n_train=len(tr)))
    R = pd.DataFrame(rec); resB[tgt] = R
    save(R, f"splits_scenarioB_{tgt}", False)
    save(R.groupby(["f", "model"])[["mae", "rmse", "bias", "rmse_sub", "mae_sub", "n_train"]].mean().round(3), f"raw_scenarioB_{tgt}")

def table4():
    out = {}
    for tgt in ["2023-11", "2023-06"]:
        R = resB[tgt]; R = R[R.f == 1.0]
        g = R.groupby("model")[["mae", "rmse", "bias"]].mean(); pr = g.loc["pers", "rmse"]
        g["skill"] = 1 - (g.rmse / pr) ** 2; out[tgt] = g
    order = ["pers", "sn", "sn_dist", "shift", "ridge", "rf", "gb"]
    T = pd.concat({("Nov 2023" if k == "2023-11" else "Jun 2023"): v.loc[order] for k, v in out.items()}, axis=1)
    T.index = [NAMES[i] for i in order]; return T.round(2)
save(table4(), "table4_gauged_accuracy")

def table4b():
    out = {}
    for tgt in ["2023-11", "2023-06"]:
        R = resB[tgt]; R = R[R.f == 1.0]
        out[("Nov 2023" if tgt == "2023-11" else "Jun 2023")] = R.groupby("model")[["mae_sub", "rmse_sub"]].mean().rename(columns=lambda c: c[:-4])
    order = ["pers", "sn", "sn_dist", "ridge", "rf", "gb"]
    T = pd.concat({k: v.loc[order] for k, v in out.items()}, axis=1); T.index = [NAMES[i] for i in order]
    n = {k: (resB[t][resB[t].f == 1].n_sub.mean(), resB[t].n_test.iloc[0]) for k, t in [("Nov 2023", "2023-11"), ("Jun 2023", "2023-06")]}
    log(f"Table 4b sample: {n}"); return T.round(2)
save(table4b(), "table4b_same_wells_subset")

# win rates of boosting over persistence / seasonal-naive (per split, full network)
wr = {}
for tgt in resB:
    R = resB[tgt][resB[tgt].f == 1.0].pivot(index="split", columns="model", values="rmse")
    Rs = resB[tgt][resB[tgt].f == 1.0].pivot(index="split", columns="model", values="rmse_sub")
    wr[tgt] = dict(gb_beats_pers=int((R.gb < R.pers).sum()), gb_beats_sn_subset=int((Rs.gb < Rs.sn).sum()), splits=len(R),
                   gb_gain_vs_sn_subset_pct=round(100 * (1 - Rs.gb.mean() / Rs.sn.mean()), 1),
                   gb_gain_vs_pers_pct=round(100 * (1 - R.gb.mean() / R.pers.mean()), 1))
save(pd.DataFrame(wr).T, "win_rates_gauged")

def thinning_table(tgt):
    R = resB[tgt]; T = R.pivot_table(index="f", columns="model", values="rmse", aggfunc="mean")[["pers", "shift", "ridge", "rf", "gb"]]
    T.columns = [NAMES[c] for c in T.columns]; T.insert(0, "Training wells (approx.)", R.groupby("f").n_train.mean().round(0).astype(int))
    return T.round(2)
save(thinning_table("2023-11"), "table5_thinning_gauged_nov")
save(thinning_table("2023-06"), "tableA1_thinning_gauged_jun")
# thinning wins vs persistence at 10% retention
w10 = {}
for tgt in resB:
    R = resB[tgt]; R = R[R.f == 0.10].pivot(index="split", columns="model", values="rmse")
    w10[tgt] = {m: int((R[m] < R.pers).sum()) for m in ["ridge", "gb", "rf"]}
save(pd.DataFrame(w10).T, "wins_vs_persistence_10pct")

# ------------------------------------------------------------------ Scenario A: Tables 6, A2 (+ district RMSE)
def predict_A(tr, te, seed):
    g = tr.y.mean(); grp = tr.groupby("district").y.agg(["sum", "count"])
    dm = te.district.map(lambda x: grp.loc[x, "sum"] / grp.loc[x, "count"] if x in grp.index else g).values
    sh = te.district.map(lambda x: (grp.loc[x, "sum"] + 3 * g) / (grp.loc[x, "count"] + 3) if x in grp.index else g).values
    # boosting on district statistics: leave-one-out district mean for training wells
    cnt_tr = tr.district.map(grp["count"]).values; sum_tr = tr.district.map(grp["sum"]).values
    loo = np.where(cnt_tr > 1, (sum_tr - tr.y.values) / np.maximum(cnt_tr - 1, 1), g)
    cnt_te = te.district.map(lambda x: grp.loc[x, "count"] if x in grp.index else 0).values
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(tr[["district", "type"]])
    Xtr = np.hstack([loo[:, None], cnt_tr[:, None], oh.transform(tr[["district", "type"]])])
    Xte = np.hstack([dm[:, None], cnt_te[:, None], oh.transform(te[["district", "type"]])])
    return {"global": np.full(len(te), g), "district": dm, "shrunk": sh, "gb": gb(seed).fit(Xtr, tr.y).predict(Xte)}

A_names = {"global": "Global mean", "district": "District mean", "shrunk": "Shrunk district mean", "gb": "Boosting (district stats)"}
distA = []
for tgt in ["2023-11", "2023-06"]:
    t = d[d[tgt].notna()].copy(); t["y"] = t[tgt]
    log(f"Scenario A {tgt}: {len(t)} wells with a reading")
    rs = np.random.RandomState(SEED + 10 + (0 if tgt == "2023-11" else 1)); rec = []
    for s in range(N_SPLITS):
        te_i, pool_i = split(len(t), rs); te = t.iloc[te_i]; yobs = te.y.values
        for f in RETAIN:
            tr = t.iloc[thin(pool_i, f, rs)]; p = predict_A(tr, te, SEED + s)
            for k, v in p.items():
                e = v - yobs
                rec.append(dict(split=s, f=f, model=k, rmse=rmse(e), mae=np.abs(e).mean(),
                                nse=1 - np.sum(e ** 2) / np.sum((yobs - yobs.mean()) ** 2), n_train=len(tr)))
                if k == "district" and f in (1.0, 0.25):
                    distA.append(pd.DataFrame(dict(target=tgt, f=f, district=te.district.values, se=e ** 2)))
    R = pd.DataFrame(rec); save(R, f"splits_scenarioA_{tgt}", False)
    T = R.pivot_table(index="f", columns="model", values="rmse", aggfunc="mean")[list(A_names)]
    T.columns = [A_names[c] for c in T.columns]; T.insert(0, "Training wells (approx.)", R.groupby("f").n_train.mean().round(0).astype(int))
    save(T.round(2), "table6_ungauged_nov" if tgt == "2023-11" else "tableA2_ungauged_jun")
    save(R[(R.f == 1.0)].groupby("model")[["rmse", "mae", "nse"]].mean().round(2), f"scenarioA_full_network_{tgt}")
D = pd.concat(distA); DD = D.groupby(["target", "f", "district"]).se.agg(rmse=lambda s: np.sqrt(s.mean()), n_test="count").round(1)
save(DD, "fig_data_district_rmse_ungauged")

# ------------------------------------------------------------------ Table 7: value of history
rows = []
variants = [("Jun-23 only", ["2023-06"], [], "2023-06"), ("+ Nov-22", ["2023-06"], ["2022-11"], "2023-06"),
            ("+ Jun-22", ["2023-06"], ["2022-11", "2022-06"], "2023-06"),
            ("+ 2019 rounds (all)", ["2023-06"], ["2022-11", "2022-06", "2019-10", "2019-06"], "2023-06"),
            ("Prior years only (no 2023 data)", ["2022-11", "2022-06", "2019-10", "2019-06"], ["2022-11", "2022-06", "2019-10", "2019-06"], None)]
for lab, anc, fh, _ in variants:
    t, feats = panel_B("2023-11", anc, fh); rs = np.random.RandomState(SEED + 20); res = {k: [] for k in ["pers", "shift", "ridge", "gb"]}
    for s in range(N_SPLITS):
        te_i, pool_i = split(len(t), rs); te, tr = t.iloc[te_i], t.iloc[pool_i]
        p = predict_B(tr, te, feats, "2023-11", None, ["shift", "ridge", "gb"], SEED + s)
        for k in res: res[k].append(rmse(p[k] - te["2023-11"].values))
    rows.append([lab, len(t)] + [np.mean(res[k]) for k in ["pers", "shift", "ridge", "gb"]])
save(pd.DataFrame(rows, columns=["History available", "Wells", "Persistence", "District-shift", "Ridge", "Boosting"]).round(2), "table7_value_of_history", False)

# ------------------------------------------------------------------ Table 8: uncertainty (bootstrap vs conformal)
def crps_gauss(y, m, s):
    z = (y - m) / s; return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
def crps_emp(y, samples):
    x = np.sort(samples); n = len(x); term2 = 2 * np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / n ** 2 / 2
    return np.mean(np.abs(x[None, :] - y[:, None]), axis=1) - term2
LEVELS = [0.5, 0.7, 0.9, 0.95]; uq = []
def qgb(alpha, seed): return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_iter=80, learning_rate=0.06, max_depth=3,
                                                         min_samples_leaf=8, l2_regularization=1.0, random_state=seed)
def iscore(y, l, u, alpha=0.10): return np.mean((u - l) + (2 / alpha) * np.maximum(l - y, 0) + (2 / alpha) * np.maximum(y - u, 0))
for tgt in ["2023-11", "2023-06"]:
    t, feats = panel_B(tgt, HIST[tgt], HIST[tgt]); log(f"Uncertainty {tgt}")
    rs = np.random.RandomState(SEED + 30 + (0 if tgt == "2023-11" else 1))
    for s in range(N_UQ):
        te_i, pool_i = split(len(t), rs); te = t.iloc[te_i]; yte = te.y.values
        for f in [1.0, 0.5, 0.25, 0.10]:
            tr = t.iloc[thin(pool_i, f, rs)]; _, _, Zg = design(tr, feats); Xtr, Xte = Zg(tr), Zg(te); ytr = tr.y.values
            point = gb(s).fit(Xtr, ytr).predict(Xte)
            # (1) bootstrap ensemble (15 models, Gaussian intervals)
            ens = np.array([gb(b).fit(Xtr[ix], ytr[ix]).predict(Xte) for b in range(15) for ix in [rs.randint(0, len(ytr), len(ytr))]])
            em, es = ens.mean(0), np.maximum(ens.std(0, ddof=1), 1e-3)
            # cross-validated out-of-fold residuals (used by methods 2 and 3)
            oof = np.zeros(len(ytr))
            for a, b in KFold(5, shuffle=True, random_state=s).split(Xtr): oof[b] = ytr[b] - gb(s).fit(Xtr[a], ytr[a]).predict(Xtr[b])
            ar = np.abs(oof); n = len(ar); sd_cv = max(oof.std(ddof=1), 1e-3)
            # (4) quantile boosting (5% and 95% quantiles, 90% interval only)
            qlo = qgb(0.05, s).fit(Xtr, ytr).predict(Xte); qhi = qgb(0.95, s).fit(Xtr, ytr).predict(Xte)
            qlo, qhi = np.minimum(qlo, qhi), np.maximum(qlo, qhi)
            for lev in LEVELS:
                z = norm.ppf(0.5 + lev / 2); q = np.quantile(ar, min(1, np.ceil((n + 1) * lev) / n))
                rec = dict(target=tgt, f=f, level=lev, rep=s,
                           picp_ens=np.mean(np.abs(yte - em) <= z * es), picp_conf=np.mean(np.abs(yte - point) <= q),
                           picp_gcv=np.mean(np.abs(yte - point) <= z * sd_cv),
                           width_ens=2 * z * es.mean(), width_conf=2 * q, width_gcv=2 * z * sd_cv)
                if lev == 0.9:
                    rec.update(picp_qgb=np.mean((yte >= qlo) & (yte <= qhi)), width_qgb=np.mean(qhi - qlo),
                               is_ens=iscore(yte, em - z * es, em + z * es), is_conf=iscore(yte, point - q, point + q),
                               is_gcv=iscore(yte, point - z * sd_cv, point + z * sd_cv), is_qgb=iscore(yte, qlo, qhi))
                uq.append(rec)
            uq.append(dict(target=tgt, f=f, level="crps", rep=s, crps_ens=np.mean(crps_gauss(yte, em, es)),
                           crps_gcv=np.mean(crps_gauss(yte, point, sd_cv)),
                           crps_conf=np.mean([crps_emp(np.array([y]), point[i] + oof)[0] for i, y in enumerate(yte)]),
                           mae=np.mean(np.abs(yte - point))))
U = pd.DataFrame(uq); save(U, "raw_uncertainty", False)
c = U[U.level == "crps"].groupby(["target", "f"])[["crps_ens", "crps_gcv", "crps_conf", "mae"]].mean()
p9 = U[U.level == 0.9].groupby(["target", "f"])[["picp_ens", "picp_gcv", "picp_qgb", "picp_conf", "width_ens", "width_gcv", "width_qgb", "width_conf",
                                                 "is_ens", "is_gcv", "is_qgb", "is_conf"]].mean()
T8w = p9.join(c).reset_index(); T8w["target"] = T8w.target.map({"2023-11": "Nov 2023", "2023-06": "Jun 2023"}); T8w["f"] = (T8w.f * 100).astype(int)
save(T8w.sort_values(["target", "f"], ascending=[False, False]).round(2), "table8_uncertainty_wide", False)
M = {"ens": "Bootstrap ensemble", "gcv": "Gaussian (CV residual SD)", "qgb": "Quantile boosting", "conf": "Conformal (CV residuals)"}
long = [dict(Target=r.target, Retained=f"{r.f}%", Method=M[k], PICP=r[f"picp_{k}"], Width=r[f"width_{k}"], IntervalScore=r[f"is_{k}"])
        for _, r in T8w.sort_values(["target", "f"], ascending=[False, False]).iterrows() for k in M]
save(pd.DataFrame(long).round(2), "table8_uncertainty", False)
rel = U[(U.level != "crps")].groupby(["target", "f", "level"])[["picp_ens", "picp_gcv", "picp_conf"]].mean().round(3)
save(rel, "fig_data_reliability_all_levels")
log("DONE - all tables written to " + OUT)
