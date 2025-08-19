#!/usr/bin/env python3
import argparse, os, sys, logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from core import ensure_dir

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.grid": True,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.frameon": True,
    "legend.fontsize": 9,
    "figure.constrained_layout.use": True
})

def setup_logging(logfile):
    logger = logging.getLogger("microgrid")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); sh.setLevel(logging.INFO)
    ensure_dir(os.path.dirname(logfile))
    fh = RotatingFileHandler(logfile, maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    logger.handlers.clear(); logger.addHandler(sh); logger.addHandler(fh)
    return logger

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.nanmean((a-b)**2)))

def detection_delay(plant, twin, thresh=0.05, window=5):
    df = plant.merge(twin, on=["timestamp","scenario"], how="inner")
    err = np.abs(df["P_load_estimated"] - df["P_load_true"]) / np.maximum(1e-6, np.abs(df["P_load_true"]))
    flag = df["event"].values.astype(bool)
    if not flag.any(): return np.nan
    idx0 = np.where(flag)[0][0]
    ebin = pd.Series(err).rolling(window, min_periods=window).mean().values
    idx = np.where(ebin[idx0:] > thresh)[0]
    return np.nan if len(idx)==0 else float(df["timestamp"].iloc[idx0 + idx[0]] - df["timestamp"].iloc[idx0])

def load_tables(data_dir, results_dir, scenarios, runs, seed0):
    plants, twins = [], []
    for sc in scenarios:
        for r in range(runs):
            seed = seed0 + r
            p = os.path.join(data_dir, sc, f"plant_data_{seed}.csv")
            t = os.path.join(results_dir, sc, f"twin_outputs_{seed}.csv")
            if os.path.exists(p) and os.path.exists(t):
                dfp = pd.read_csv(p); dft = pd.read_csv(t)
                plants.append(dfp); twins.append(dft)
    return (pd.concat(plants, ignore_index=True) if plants else pd.DataFrame(),
            pd.concat(twins, ignore_index=True) if twins else pd.DataFrame())

def per_run_metrics(plant, twin):
    rows = []
    keys = ["scenario"]
    for (sc,), g in tqdm(plant.groupby(keys), desc="metrics:per-scenario"):
        tg = twin[twin["scenario"]==sc]
        if tg.empty: continue
        df = g.merge(tg, on=["timestamp","scenario"], how="inner")
        rows.append({
            "scenario": sc,
            "rmse_V1": rmse(df["V_bus1_true"], df["V_bus1_model"]),
            "rmse_V2": rmse(df["V_bus2_true"], df["V_bus2_model"]),
            "rmse_V3": rmse(df["V_bus3_true"], df["V_bus3_model"]),
            "rmse_Pload": rmse(df["P_load_true"], df["P_load_estimated"]),
            "delay_s": detection_delay(g, tg)
        })
    return pd.DataFrame(rows)

def plot_timeseries(plant, twin, scenario, outdir):
    dfp = plant[plant["scenario"]==scenario].copy()
    dft = twin[twin["scenario"]==scenario].copy()
    if dfp.empty or dft.empty: return
    df = dfp.merge(dft, on=["timestamp","scenario"], how="inner")

    t = df["timestamp"].values
    event = df["event"].values.astype(bool)
    event_spans = _spans_from_mask(t, event)

    fig = plt.figure(figsize=(10.5, 9.0), layout="constrained")
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[2,0])
    ax4 = fig.add_subplot(gs[3,0])

    ax1.plot(t, df["V_bus1_true"], label="V1 True")
    ax1.plot(t, df["V_bus1_model"], "--", label="V1 Twin")
    ax1.set_ylabel("Voltage (p.u.)")
    ax1.set_title(f"Scenario: {scenario} — Bus Voltages & Load Estimation")

    ax2.plot(t, df["V_bus2_true"], label="V2 True")
    ax2.plot(t, df["V_bus2_model"], "--", label="V2 Twin")
    ax2.set_ylabel("Voltage (p.u.)")

    ax3.plot(t, df["V_bus3_true"], label="V3 True")
    ax3.plot(t, df["V_bus3_model"], "--", label="V3 Twin")
    ax3.set_ylabel("Voltage (p.u.)")

    ax4.plot(t, df["P_load_true"], label="Load True")
    ax4.plot(t, df["P_load_estimated"], "--", label="Load EKF")
    if "EKF_P_load_uncertainty" in df:
        s2 = np.maximum(0, df["EKF_P_load_uncertainty"].values)
        s = np.sqrt(s2)
        if np.all(np.isfinite(s)):
            ax4.fill_between(t,
                             df["P_load_estimated"] - 1.96*s,
                             df["P_load_estimated"] + 1.96*s,
                             alpha=0.2, label="EKF 95% CI")
    ax4.set_ylabel("Power (MW)")
    ax4.set_xlabel("Time (s)")

    for ax in [ax1, ax2, ax3, ax4]:
        for (a,b) in event_spans:
            ax.axvspan(a, b, color="grey", alpha=0.15, lw=0)

    ax1.legend(ncol=2); ax2.legend(ncol=2); ax3.legend(ncol=2); ax4.legend(ncol=2)
    ensure_dir(outdir)
    fpng = os.path.join(outdir, f"timeseries_{scenario}.png")
    fpdf = os.path.join(outdir, f"timeseries_{scenario}.pdf")
    fig.savefig(fpng, bbox_inches="tight")
    fig.savefig(fpdf, bbox_inches="tight")
    plt.close(fig)

def _spans_from_mask(t, m):
    spans = []
    if len(t)==0: return spans
    in_span = False; start = None
    for i,flag in enumerate(m):
        if flag and not in_span: in_span=True; start=t[i]
        if not flag and in_span: in_span=False; spans.append((start, t[i]))
    if in_span: spans.append((start, t[-1]))
    return spans

def plot_metric_table(metrics_df, outdir):
    ensure_dir(outdir)
    csvp = os.path.join(outdir, "summary_metrics.csv")
    metrics_df.to_csv(csvp, index=False)
    try:
        latex = (metrics_df
                 .set_index("scenario")
                 .rename(columns={"rmse_V1":"RMSE V1","rmse_V2":"RMSE V2","rmse_V3":"RMSE V3",
                                  "rmse_Pload":"RMSE Load","delay_s":"Detect Delay (s)"})
                 .to_latex(float_format="%.5f"))
        with open(os.path.join(outdir, "summary_metrics.tex"), "w") as f: f.write(latex)
    except Exception as e:
        logging.getLogger("microgrid").warning(f"LaTeX export skipped: {e}")

def plot_rmse_bar(metrics_df, outdir):
    ensure_dir(outdir)
    fig = plt.figure(figsize=(9,4.2))
    ax = fig.add_subplot(111)
    idx = np.arange(len(metrics_df))
    width = 0.18
    ax.bar(idx-1.5*width, metrics_df["rmse_V1"], width, label="V1")
    ax.bar(idx-0.5*width, metrics_df["rmse_V2"], width, label="V2")
    ax.bar(idx+0.5*width, metrics_df["rmse_V3"], width, label="V3")
    ax.bar(idx+1.5*width, metrics_df["rmse_Pload"], width, label="Load")
    ax.set_xticks(idx); ax.set_xticklabels(metrics_df["scenario"], rotation=15)
    ax.set_ylabel("RMSE"); ax.set_title("RMSE by Scenario")
    ax.legend(ncol=4)
    fig.savefig(os.path.join(outdir, "rmse_bar.png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "rmse_bar.pdf"), bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--fig_dir", default="paper/figures")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--scenarios", nargs="+", default=[
        "baseline","pv_ramp","load_step","sensor_fault","grid_sag","line_outage","sensor_bias","noise_burst"
    ])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--log", default="logs/analysis.log")
    args = ap.parse_args()

    setup_logging(args.log)
    logger = logging.getLogger("microgrid")

    plant, twin = load_tables(args.data_dir, args.results_dir, args.scenarios, args.runs, args.seed0)
    if plant.empty or twin.empty:
        logger.error("No data found. Run experiments first."); sys.exit(1)

    metrics = per_run_metrics(plant, twin).sort_values("scenario")
    plot_metric_table(metrics, args.out_dir)
    plot_rmse_bar(metrics, args.fig_dir)

    try:
        for sc in args.scenarios:
            plot_timeseries(plant, twin, sc, args.fig_dir)
    except KeyboardInterrupt:
        logger.info("Interrupted by user; partial outputs saved.")
        sys.exit(130)

    logger.info(f"Analysis complete. CSV/TEX in {args.out_dir}, figures in {args.fig_dir}")

if __name__ == "__main__":
    main()
