#!/usr/bin/env python3
import argparse, logging, sys, os
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
from core import SimConfig, TwinConfig, generate_plant_data, run_digital_twin, ensure_dir
import pandas as pd

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

def run_experiments(scenarios, runs, T, dt, seed0, data_dir, results_dir):
    logger = logging.getLogger("microgrid")
    tcfg = TwinConfig()
    ensure_dir(data_dir); ensure_dir(results_dir)

    for sc in scenarios:
        logger.info(f"Scenario: {sc}")
        for r in tqdm(range(runs), desc=f"runs:{sc}"):
            seed = seed0 + r
            scfg = SimConfig(T=T, dt=dt, seed=seed, scenario=sc, out_dir=data_dir)
            plant_df = generate_plant_data(scfg)
            ddir = os.path.join(data_dir, sc); ensure_dir(ddir)
            p_path = os.path.join(ddir, f"plant_data_{seed}.csv")
            plant_df.to_csv(p_path, index=False)

            twin_df = run_digital_twin(plant_df, tcfg)
            rdir = os.path.join(results_dir, sc); ensure_dir(rdir)
            t_path = os.path.join(rdir, f"twin_outputs_{seed}.csv")
            twin_df.to_csv(t_path, index=False)

            logger.info(f"Saved: {p_path} | {t_path} | rows={len(twin_df)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=[
        "baseline","pv_ramp","load_step","sensor_fault","grid_sag","line_outage","sensor_bias","noise_burst"
    ])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--T", type=int, default=3600)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--log", default="logs/experiments.log")
    args = ap.parse_args()

    setup_logging(args.log)
    run_experiments(args.scenarios, args.runs, args.T, args.dt, args.seed0, args.data_dir, args.results_dir)

if __name__ == "__main__":
    main()
