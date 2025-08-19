#!/usr/bin/env python3
import os, math, logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower import LoadflowNotConverged
from tqdm import tqdm

logger = logging.getLogger("microgrid")

# ----------------------------- Network ---------------------------------
def create_microgrid_network():
    net = pp.create_empty_network()
    b_gen = pp.create_bus(net, vn_kv=11.0, name='GenBus')
    b_load = pp.create_bus(net, vn_kv=11.0, name='LoadBus')
    b_pv = pp.create_bus(net, vn_kv=11.0, name='PVBus')

    pp.create_line_from_parameters(net, b_gen, b_load, 1.0, 0.1, 0.4, 10, 0.4, name="L1")
    pp.create_line_from_parameters(net, b_load, b_pv, 1.0, 0.1, 0.4, 10, 0.4, name="L2")

    pp.create_ext_grid(net, bus=b_gen, vm_pu=1.02, name='Slack')
    pp.create_gen(net, bus=b_gen, p_mw=5.0, vm_pu=1.02, name='SyncGen')
    pp.create_sgen(net, bus=b_pv, p_mw=3.0, q_mvar=0.0, name='PV')
    pp.create_load(net, bus=b_load, p_mw=6.0, q_mvar=1.0, name='Load')
    return net, {"GenBus": b_gen, "LoadBus": b_load, "PVBus": b_pv}

def run_pf(net):
    try:
        pp.runpp(net, algorithm="nr", init="auto", enforce_q_lims=True, run_control=False, tolerance_mva=1e-6)
        return True
    except LoadflowNotConverged:
        try:
            pp.runpp(net, algorithm="bfsw", init="flat", run_control=False)
            return True
        except LoadflowNotConverged:
            return False

# ------------------------------ EKF ------------------------------------
def numeric_jacobian(func, x, *args, epsilon=1e-6):
    x = np.asarray(x, dtype=float)
    y0 = np.atleast_1d(func(x, *args))
    m, n = y0.size, x.size
    J = np.zeros((m, n))
    for i in range(n):
        xp = x.copy(); xm = x.copy()
        xp[i] += epsilon; xm[i] -= epsilon
        J[:, i] = (np.atleast_1d(func(xp, *args)) - np.atleast_1d(func(xm, *args))) / (2*epsilon)
    return J

class SimpleEKF:
    def __init__(self, x0, P0, Q, R):
        self.x = np.asarray(x0, float)
        self.P = np.asarray(P0, float)
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.I = np.eye(len(x0))

    def predict(self, f, u=None, dt=1.0):
        self.x = np.atleast_1d(f(self.x, u, dt))
        F = numeric_jacobian(f, self.x, u, dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, h, z):
        H = numeric_jacobian(h, self.x)
        y = np.asarray(z) - np.atleast_1d(h(self.x))
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

# ----------------------- Simulator + Twin -------------------------------
@dataclass
class SimConfig:
    T: int = 3600
    dt: float = 1.0
    seed: int = 42
    scenario: str = "baseline"
    out_dir: str = "data"

def _build_scenario_masks(time_vector, cfg: SimConfig):
    t = time_vector
    zeros = np.zeros_like(t, dtype=bool)
    event = zeros.copy()

    masks = {
        "pv_ramp": zeros.copy(),
        "load_step": zeros.copy(),
        "sensor_fault": zeros.copy(),
        "grid_sag": zeros.copy(),
        "line_outage": zeros.copy(),
        "sensor_bias": zeros.copy(),
        "noise_burst": zeros.copy(),
    }

    if cfg.scenario == "pv_ramp":
        t0, dur = 1200, 60
        m = (t >= t0) & (t < t0 + dur); masks["pv_ramp"] = m; event |= m
    elif cfg.scenario == "load_step":
        t0, dur = 1800, 120
        m = (t >= t0) & (t < t0 + dur); masks["load_step"] = m; event |= m
    elif cfg.scenario == "sensor_fault":
        t0, dur = 2400, 120
        m = (t >= t0) & (t < t0 + dur); masks["sensor_fault"] = m; event |= m
    elif cfg.scenario == "grid_sag":
        t0, dur = 900, 90
        m = (t >= t0) & (t < t0 + dur); masks["grid_sag"] = m; event |= m
    elif cfg.scenario == "line_outage":
        t0, dur = 1500, 120
        m = (t >= t0) & (t < t0 + dur); masks["line_outage"] = m; event |= m
    elif cfg.scenario == "sensor_bias":
        t0, dur = 2100, 180
        m = (t >= t0) & (t < t0 + dur); masks["sensor_bias"] = m; event |= m
    elif cfg.scenario == "noise_burst":
        t0, dur = 2700, 90
        m = (t >= t0) & (t < t0 + dur); masks["noise_burst"] = m; event |= m

    return masks, event

def generate_plant_data(cfg: SimConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    net, buses = create_microgrid_network()
    t = np.arange(0, cfg.T, cfg.dt)
    masks, event = _build_scenario_masks(t, cfg)

    G_peak = 1000
    P_rated_pv = 3.0
    G_base = np.maximum(0, G_peak * np.sin(np.pi * (t/3600) / 12.0))
    cloud_mask = np.ones_like(t, dtype=float)
    N_clouds = rng.poisson(6)
    for _ in range(N_clouds):
        t0 = rng.uniform(0, cfg.T); depth = rng.uniform(0.2, 0.8); dur = rng.uniform(30, 120)
        s = np.searchsorted(t, t0); e = np.searchsorted(t, t0 + dur)
        cloud_mask[s:e] *= (1 - depth)
    G = np.clip(G_base * cloud_mask, 0, G_peak)
    P_pv_true = P_rated_pv * (G / 1000.0)

    P_load_base = 6.0
    P_load_true = P_load_base + rng.normal(0, 0.02 * P_load_base, size=t.size)

    if masks["pv_ramp"].any():
        idx = np.where(masks["pv_ramp"])[0]
        r = np.linspace(1.0, 0.5, idx.size)
        P_pv_true[idx] *= r
    if masks["load_step"].any():
        P_load_true[masks["load_step"]] *= 1.2

    sigma_v_base, sigma_p = 0.002, 0.01
    sigma_v = np.full(t.size, sigma_v_base)
    if masks["noise_burst"].any():
        sigma_v[masks["noise_burst"]] = 0.01

    stuck_val = None
    data = []

    for k in tqdm(range(t.size), desc=f"simulate:{cfg.scenario}", leave=False):
        # align voltage controllers on the GenBus
        vm = 0.97 if masks["grid_sag"][k] else 1.02
        net.ext_grid.at[0, "vm_pu"] = vm
        if len(net.gen):
            net.gen.at[0, "vm_pu"] = vm

        # line outage toggle for L2
        if masks["line_outage"][k]:
            net.line.in_service.iloc[1] = False
        else:
            net.line.in_service.iloc[1] = True

        # apply true injections
        net.sgen.at[0, 'p_mw'] = float(P_pv_true[k])
        net.load.at[0, 'p_mw'] = float(P_load_true[k])

        # power flow
        if not run_pf(net):
            logger.warning(f"PF failed at t={t[k]:.1f}s"); continue

        V_true = net.res_bus.vm_pu.values
        P_gen_true = net.res_gen.p_mw.values[0]

        # measurements
        v_noise = np.array([rng.normal(0, sigma_v[k]) for _ in range(3)])
        V_meas = V_true + v_noise
        if masks["sensor_bias"][k]:
            V_meas[1] += 0.01

        P_pv_meas = P_pv_true[k] + rng.normal(0, sigma_p)
        if masks["sensor_fault"][k]:
            if stuck_val is None: stuck_val = P_pv_meas
            P_pv_meas = stuck_val
        else:
            stuck_val = None

        P_gen_meas = P_gen_true + rng.normal(0, sigma_p)

        data.append([
            t[k], cfg.scenario, P_pv_true[k], P_pv_meas, P_gen_true, P_gen_meas,
            V_true[0], V_meas[0], V_true[1], V_meas[1], V_true[2], V_meas[2],
            P_load_true[k], bool(event[k])
        ])


    cols = ['timestamp','scenario','P_pv_true','P_pv_meas','P_gen_true','P_gen_meas',
            'V_bus1_true','V_bus1_meas','V_bus2_true','V_bus2_meas','V_bus3_true','V_bus3_meas',
            'P_load_true','event']
    return pd.DataFrame(data, columns=cols)

@dataclass
class TwinConfig:
    sigma_v: float = 0.002
    Q_var: float = 1e-4
    P0_var: float = 100.0

def run_digital_twin(plant_df: pd.DataFrame, twin_cfg: TwinConfig) -> pd.DataFrame:
    net, buses = create_microgrid_network()

    x0 = np.array([6.0])
    P0 = np.eye(1) * twin_cfg.P0_var
    Q = np.eye(1) * twin_cfg.Q_var
    R = np.diag([twin_cfg.sigma_v**2]*3)

    ekf = SimpleEKF(x0, P0, Q, R)

    def f(x, u, dt): return x
    def h(x):
        net.load.at[0, 'p_mw'] = float(x[0])
        net.load.at[0, 'q_mvar'] = 1.0
        if not run_pf(net): return np.full(3, np.nan)
        return net.res_bus.vm_pu.values[[0,1,2]]

    out = []
    it = tqdm(plant_df.itertuples(index=False), total=len(plant_df), desc="twin", leave=False)
    for row in it:
        z = np.array([row.V_bus1_meas, row.V_bus2_meas, row.V_bus3_meas])
        if np.isnan(z).any(): continue

        net.sgen.at[0, 'p_mw'] = float(row.P_pv_meas)

        ekf.predict(f, None, 1.0)
        ekf.update(h, z)

        net.load.at[0, 'p_mw'] = float(ekf.x[0])
        if not run_pf(net): continue

        V = net.res_bus.vm_pu.values
        out.append({
            "timestamp": row.timestamp,
            "scenario": row.scenario,
            "V_bus1_model": V[0],
            "V_bus2_model": V[1],
            "V_bus3_model": V[2],
            "P_load_estimated": float(ekf.x[0]),
            "EKF_P_load_uncertainty": float(ekf.P[0,0])
        })
    return pd.DataFrame(out)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
