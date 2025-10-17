#!/usr/bin/env python3
"""
Digital Twin of an EV Charger + Battery Pack (physics-informed, ML-friendly)
-----------------------------------------------------------------------------
Key capabilities:
- Bidirectional charge/discharge with CC/CV and power/current limits
- Physics-based battery model (1-RC Thevenin) with SOC, terminal voltage, IR drop
- Lumped thermal model with I^2R + entropic heat, ambient cooling, thermal derating
- Simple aging (SOH) model: capacity fade & resistance growth vs. T, C-rate, time
- Thermal runaway risk index with instantaneous alerting
- Continuous monitoring + rule-based safety controller to extend battery life
- Clean interfaces/hooks for ML: feature extraction, surrogate models, anomaly flags

Dependencies: numpy (required), optional matplotlib for demo plotting

This file can be run directly to simulate a scenario.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# --------------------------
# Utility helpers
# --------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------
# Battery + Charger Parameters
# --------------------------
@dataclass
class BatteryParams:
    # Nominal pack values (Nexon EV class; 96s NMC, ~333 V, ~126 Ah)
    # These are realistic defaults; tweak per your target vehicle.
    capacity_Ah: float = 126.0           # pack capacity at BOL (Ah)
    n_series: int = 96                   # 96 cells in series → nominal ~333 V
    n_parallel: int = 1                  # keep 1; Ah already reflects pack

    # Electrical model (Thevenin 1-RC) — conservative but realistic magnitudes
    r0_ohm: float = 0.010                # DC ohmic resistance (pack, Ω)
    r1_ohm: float = 0.015                # polarization resistance (Ω)
    c1_f: float = 2500.0                 # polarization capacitance (F)
    # get the electrical model of the battery
    # Voltage limits (per cell, converted to pack inside model)
    v_min_cell: float = 2.8              # Li-ion (NMC/NCA) lower limit
    v_max_cell: float = 4.2              # Li-ion (NMC/NCA) upper limit

    # Thermal
    mass_kg: float = 332.0               # pack mass (kg)
    heat_capacity_j_per_kgK: float = 1000.0  # Li-ion typical cp
    h_w_per_K: float = 220.0             # effective cooling conductance (active liquid cooling)
    t_ambient_C: float = 25.0

    # Entropic heat coefficient (approx.; W per A at ref SOC band and 25C)
    entropic_w_per_A: float = -0.2       # sign: negative reduces net heat during charge

    # Aging (very simplified; coefficients tuned for demonstration only)
    cal_k_ref_per_s: float = 1e-11       # calendar fade rate at 25C
    cyc_k_ref: float = 1e-5              # cycle fade rate per (C-rate^alpha)
    alpha_cyc: float = 0.7           # C-rate exponent
    arrhenius_Ea_cal_J: float = 2.5e4    # activation energy calendar aging
    arrhenius_Ea_cyc_J: float = 2.0e4

    # Thermal runaway thresholds (illustrative)
    t_warn_C: float = 45.0
    t_hot_C: float = 55.0
    t_crit_C: float = 70.0

    # Derating bands
    derate_start_C: float = 45.0         # start reducing current above optimal band
    derate_stop_C: float = 55.0          # at/above this, current goes to 0

    # Charger limits — class-typical for 400 V SUVs (India market)
    max_charge_power_W: float = 60000.0  # 60 kW DC fast (peak)
    max_discharge_power_W: float = 50000.0  # regen/V2G ceiling (conservative)
    max_c_rate: float = 0.5         # <= 1C at cool temps keeps thermal sane


@dataclass
class ChargerParams:
    cc_current_A: float = 150.0          # default CC setpoint (amps)
    # 4.10 V/cell CV target (longevity-friendly): 4.10 * 96 = 393.6 V
    cv_voltage_V: float = 4.10 * 96.0
    enable_discharge: bool = False


# --------------------------
# Core Battery Model
# --------------------------
class BatteryModel:
    def __init__(self, p: BatteryParams):
        self.p = p
        self.capacity_Ah0 = p.capacity_Ah
        self.capacity_Ah = p.capacity_Ah  # degrades over time
        self.soh = 1.0                    # state of health (capacity fraction)
        self.soc = 0.5                    # 0..1
        self.v1 = 0.0                     # RC polarization state (V)
        self.temp_C = p.t_ambient_C
        self.r0 = p.r0_ohm
        self.r1 = p.r1_ohm
        self.elapsed_s = 0.0              # track simulation time (s)
        self.throughput_As = 0.0          # ampere-seconds throughput for EFC estimate
        self.ocv_table_soc = np.linspace(0, 1, 101)
        # Simple OCV curve (pack); use better maps if available
        cell_ocv = 3.0 + 1.2 * self.ocv_table_soc - 0.1 * np.sin(5 * self.ocv_table_soc)
        self.ocv_table_V = cell_ocv * p.n_series

        # Precompute pack limits
        self.v_min_pack = p.v_min_cell * p.n_series
        self.v_max_pack = p.v_max_cell * p.n_series

    def ocv(self, soc: float, temp_C: float) -> float:
        soc_clamped = clamp(soc, 0.0, 1.0)
        return float(np.interp(soc_clamped, self.ocv_table_soc, self.ocv_table_V))

    def r_total(self) -> float:
        # Simple SOH-linked resistance growth (can be learned/updated)
        growth = (1.0 / self.soh) - 1.0
        return self.p.r0_ohm * (1.0 + 0.8 * growth) + self.p.r1_ohm

    def capacity_As(self) -> float:
        return self.capacity_Ah * 3600.0

    def step(self, i_pack_A: float, dt_s: float) -> Dict[str, float]:
        """Advance the battery states by dt_s under pack current i_pack_A.
        Positive current = discharge (convention common in battery modeling).
        """
        p = self.p
        # Electrical dynamics (Thevenin 1-RC)
        # dV1/dt = -V1/(R1*C1) + I/C1
        self.v1 += dt_s * (-self.v1 / (p.r1_ohm * p.c1_f) + i_pack_A / p.c1_f)

        ocv = self.ocv(self.soc, self.temp_C)
        v_term = ocv - i_pack_A * p.r0_ohm - self.v1

        # SOC dynamics: dSOC/dt = -I / (Capacity)
        self.soc += dt_s * (-i_pack_A / self.capacity_As())
        self.soc = float(clamp(self.soc, 0.0, 1.0))
        # Time & throughput bookkeeping
        self.elapsed_s += dt_s
        self.throughput_As += abs(i_pack_A) * dt_s

        # Thermal dynamics (lumped)
        r_tot = self.r_total()
        q_joule = (i_pack_A ** 2) * r_tot  # W
        q_entropic = p.entropic_w_per_A * i_pack_A  # W, crude sign convention
        q_in = q_joule + q_entropic
        q_out = p.h_w_per_K * (self.temp_C - p.t_ambient_C)
        dT = (q_in - q_out) / (p.mass_kg * p.heat_capacity_j_per_kgK)
        self.temp_C += dt_s * dT

        # Simple aging update
        self._update_aging(i_pack_A, dt_s)

        # Risk index
        risk = self.thermal_runaway_risk(self.temp_C, dT)

        return {
            "v_term": v_term,
            "ocv": ocv,
            "soc": self.soc,
            "temp_C": self.temp_C,
            "soh": self.soh,
            "risk": risk,
            "elapsed_s": self.elapsed_s,
        }

    def _arrhenius(self, Ea_J: float, temp_C: float) -> float:
        R = 8.314
        T_K = temp_C + 273.15
        return math.exp(-Ea_J / (R * T_K))

    def _update_aging(self, i_A: float, dt_s: float) -> None:
        p = self.p
        # Calendar fade
        k_cal = p.cal_k_ref_per_s * self._arrhenius(p.arrhenius_Ea_cal_J, self.temp_C)
        dQ_cal = k_cal * dt_s
        # Cycle fade ~ k * (|C-rate|^alpha)
        c_rate = abs(i_A) / max(self.capacity_Ah, 1e-6)
        k_cyc = p.cyc_k_ref * self._arrhenius(p.arrhenius_Ea_cyc_J, self.temp_C)
        dQ_cyc = k_cyc * (c_rate ** p.alpha_cyc) * dt_s
        # Update SOH and capacity; clamp to minimum SOH
        loss = dQ_cal + dQ_cyc
        self.capacity_Ah = max(0.6 * self.capacity_Ah0, self.capacity_Ah0 * (1.0 - loss))
        self.soh = self.capacity_Ah / self.capacity_Ah0

    # Convenience: expose elapsed time in days/hours (for ML targets or logs)
    def elapsed_days_hours(self) -> Tuple[int, int, float]:
        days = int(self.elapsed_s // 86400)
        hours = int((self.elapsed_s % 86400) // 3600)
        rem_s = float(self.elapsed_s % 3600)
        return days, hours, rem_s

    # Approximate equivalent full cycles (EFC) counter from throughput
    def equivalent_full_cycles(self) -> float:
        return self.throughput_As / max(self.capacity_As(), 1e-6)

    def thermal_runaway_risk(self, temp_C: float, dT_dt: float) -> float:
        # Composite risk: temperature proximity + heating rate
        x = (temp_C - self.p.t_crit_C) / 5.0 + 3.0 * dT_dt
        return sigmoid(x)


# --------------------------
# Charger Controller (Safety + CC/CV + Derating)
# --------------------------
@dataclass
class ControllerState:
    last_alerts: List[str] = field(default_factory=list)


class ChargerController:
    def __init__(self, b: BatteryModel, cp: ChargerParams):
        self.batt = b
        self.cp = cp
        self.state = ControllerState()

    def derate_factor(self, temp_C: float) -> float:
        p = self.batt.p
        if temp_C <= p.derate_start_C:
            return 1.0
        if temp_C >= p.derate_stop_C:
            return 0.0
        # Linear derate between start and stop
        return float((p.derate_stop_C - temp_C) / (p.derate_stop_C - p.derate_start_C))

    def compute_current_setpoint(self, mode: str, v_term: float) -> float:
        p = self.batt.p
        cp = self.cp
        cap_Ah = max(self.batt.capacity_Ah, 1e-6)

        # Base current limit by C-rate and power
        i_c_rate_limit = p.max_c_rate * cap_Ah
        if mode == "charge":
            i_power_limited = p.max_charge_power_W / max(v_term, 1e-3)
            i_limit = min(i_c_rate_limit, i_power_limited)
            i_cmd = -min(abs(cp.cc_current_A), i_limit)  # negative for charge
        elif mode == "discharge" and cp.enable_discharge:
            i_power_limited = p.max_discharge_power_W / max(v_term, 1e-3)
            i_limit = min(i_c_rate_limit, i_power_limited)
            i_cmd = +min(abs(cp.cc_current_A), i_limit)
        else:
            i_cmd = 0.0

        # Temperature derating
        der = self.derate_factor(self.batt.temp_C)
        i_cmd *= der

        # CV taper near max voltage (pack-level)
        v_cv = cp.cv_voltage_V
        if (mode == "charge") and (v_term >= v_cv):
            # Reduce current to hold voltage <= v_cv
            # Simple proportional taper towards zero
            over = v_term - v_cv
            i_cmd = -0.1 * over / max(self.batt.p.r0_ohm + 1e-6, 1e-6)

        # SOC guard rails
        if self.batt.soc >= 0.995 and i_cmd < 0:
            i_cmd = 0.0
        if self.batt.soc <= 0.005 and i_cmd > 0:
            i_cmd = 0.0

        return float(i_cmd)

    def check_alerts(self, meas: Dict[str, float]) -> List[str]:
        p = self.batt.p
        alerts = []
        if meas["temp_C"] >= p.t_warn_C:
            alerts.append("WARN: High temperature")
        if meas["temp_C"] >= p.t_hot_C:
            alerts.append("ALERT: Overtemperature shutdown")
        if meas["risk"] > 0.8:
            alerts.append("ALERT: Thermal runaway risk")
        if meas["v_term"] >= self.batt.v_max_pack:
            alerts.append("ALERT: Overvoltage")
        if meas["v_term"] <= self.batt.v_min_pack:
            alerts.append("ALERT: Undervoltage")
        if self.batt.soh < 0.75:
            alerts.append("WARN: SOH degraded")
        self.state.last_alerts = alerts
        return alerts


# --------------------------
# Digital Twin Wrapper
# --------------------------
class EVChargerTwin:
    def __init__(self, bp: BatteryParams = BatteryParams(), cp: ChargerParams = ChargerParams(), use_controller: bool = True):
        self.batt = BatteryModel(bp)
        self.ctrl = ChargerController(self.batt, cp)
        self.use_controller = use_controller
        self.batt = BatteryModel(bp)
        self.ctrl = ChargerController(self.batt, cp)
        self.history: Dict[str, List[float]] = {k: [] for k in [
            "t_s","i_A","v_V","soc","temp_C","soh","risk","mode","alerts_count","elapsed_s"
        ]}
        self.alert_log: List[Tuple[float, str]] = []

    def step(self, dt_s: float, external_mode: str = "charge") -> Dict[str, float]:
        # Measure current terminal voltage using last states (predictor step)
        v_guess = self.batt.ocv(self.batt.soc, self.batt.temp_C) - self.batt.v1
        # Determine current setpoint
        if self.use_controller:
            i_cmd = self.ctrl.compute_current_setpoint(external_mode, v_guess)
        else:
            i_cmd = self.simple_current_no_control(external_mode, v_guess)
        # Integrate physics
        meas = self.batt.step(i_cmd, dt_s)
        meas["i_A"] = i_cmd
        meas["v_V"] = meas.pop("v_term")

        # Monitoring & alerts (instantaneous)
        alerts = self.ctrl.check_alerts({**meas, "v_term": meas["v_V"]})
        for a in alerts:
            self.alert_log.append((self.history["t_s"][-1] if self.history["t_s"] else 0.0, a))

        # Log history
        t_next = (self.history["t_s"][-1] + dt_s) if self.history["t_s"] else 0.0
        self._log(t_next, external_mode, meas, len(alerts))
        return meas

    def simple_current_no_control(self, mode: str, v_term: float) -> float:
        """Open-loop current: CC with power/C-rate limits; no thermal derating, CV taper, or SOC guards.
        Negative current charges the pack; positive discharges if enabled."""
        p = self.batt.p
        cp = self.ctrl.cp
        cap_Ah = max(self.batt.capacity_Ah, 1e-6)
        i_c_rate_limit = p.max_c_rate * cap_Ah
        if mode == "charge":
            i_power_limited = p.max_charge_power_W / max(v_term, 1e-3)
            i_limit = min(i_c_rate_limit, i_power_limited)
            return -min(abs(cp.cc_current_A), i_limit)
        elif mode == "discharge" and cp.enable_discharge:
            i_power_limited = p.max_discharge_power_W / max(v_term, 1e-3)
            i_limit = min(i_c_rate_limit, i_power_limited)
            return +min(abs(cp.cc_current_A), i_limit)
        else:
            return 0.0

    def _log(self, t_s: float, mode: str, meas: Dict[str, float], n_alerts: int) -> None:
        self.history["t_s"].append(t_s)
        self.history["i_A"].append(meas["i_A"])
        self.history["v_V"].append(meas["v_V"])
        self.history["soc"].append(meas["soc"])
        self.history["temp_C"].append(meas["temp_C"])
        self.history["soh"].append(meas["soh"])
        self.history["risk"].append(meas["risk"])
        self.history["mode"].append(1.0 if mode == "charge" else -1.0)
        self.history["alerts_count"].append(float(n_alerts))
        self.history["elapsed_s"].append(meas["elapsed_s"])

    # ----------------------
    # Interfaces for ML
    # ----------------------
    def features(self, window: int = 20) -> np.ndarray:
        """Return last `window` samples as features for ML models.
        Columns: [i, v, soc, temp, soh, risk, mode]
        """
        def tail(x):
            arr = np.array(self.history[x], dtype=float)
            if len(arr) < window:
                pad = np.zeros(window - len(arr))
                arr = np.concatenate([pad, arr])
            else:
                arr = arr[-window:]
            return arr

        X = np.stack([
            tail("i_A"), tail("v_V"), tail("soc"), tail("temp_C"), tail("soh"), tail("risk"), tail("mode")
        ], axis=1)
        return X  # shape [window, 7]

    def labels_for_supervision(self) -> Dict[str, float]:
        """Targets to learn: next-step delta-T, delta-V, risk, and time markers.
        Replace with your task-specific labels.
        """
        if not self.history["t_s"]:
            d, h, rem = self.batt.elapsed_days_hours()
            return {
                "dT_next": 0.0, "dV_next": 0.0,
                "risk": float(self.batt.thermal_runaway_risk(self.batt.temp_C, 0.0)),
                "elapsed_days": float(d), "elapsed_hours": float(h) + rem/3600.0,
                "efc": self.batt.equivalent_full_cycles(),
            }
        dT = 0.0
        dV = 0.0
        if len(self.history["temp_C"]) >= 2:
            dT = self.history["temp_C"][-1] - self.history["temp_C"][-2]
        if len(self.history["v_V"]) >= 2:
            dV = self.history["v_V"][-1] - self.history["v_V"][-2]
        d, h, rem = self.batt.elapsed_days_hours()
        return {
            "dT_next": dT, "dV_next": dV,
            "risk": self.history["risk"][-1],
            "elapsed_days": float(d),
            "elapsed_hours": float(h) + rem/3600.0,
            "efc": self.batt.equivalent_full_cycles(),
        }

    # Optional: plug in an ML regressor to estimate SoH from the rolling window.
    # `estimator` should implement estimator.predict(X_flat[np.newaxis, :]) -> [soh_hat]
    def ml_estimate_soh(self, estimator: Callable[[np.ndarray], np.ndarray], window: int = 60) -> float:
        X = self.features(window=window).astype(np.float32)
        x_flat = X.flatten()
        soh_hat = float(estimator(x_flat[np.newaxis, ...])[0])
        return float(clamp(soh_hat, 0.5, 1.05))  # sanity clamp

    def anomaly_score(self) -> float:
        """Crude anomaly indicator based on unexpected dT and V drift vs. simple physics.
        In production: replace with a learned residual model.
        """
        if len(self.history["t_s"]) < 2:
            return 0.0
        # Expected small changes
        dT = self.history["temp_C"][-1] - self.history["temp_C"][-2]
        dV = self.history["v_V"][-1] - self.history["v_V"][-2]
        score = abs(dT) * 10 + abs(dV) * 0.01
        return float(clamp(score, 0.0, 1.0))


# --------------------------
# Example scenario / CLI-less demo
# --------------------------

def demo(sim_s: float = 86400.0, dt_s: float = 1.0, mode_schedule: Optional[List[Tuple[float, str]]] = None,
         plot: bool = False, use_controller: bool = True, csv_path: Optional[str] = None,
         show_progress: Optional[bool] = None,
         bp: Optional[BatteryParams] = None,
         cp: Optional[ChargerParams] = None) -> EVChargerTwin:
    twin = EVChargerTwin(bp=bp or BatteryParams(), cp=cp or ChargerParams(), use_controller=use_controller)
    # Example: start charging, then pause, then discharge for a bit if enabled
    schedule = mode_schedule or [
        (0.0, "charge"),
        (1200.0, "idle"),
        (1500.0, "charge"),
    ]
    next_idx = 1

    n_steps = int(sim_s / dt_s)
    use_tqdm = (_HAS_TQDM and (show_progress is None or show_progress)) or (_HAS_TQDM and show_progress is True)
    iterator = range(n_steps)
    if use_tqdm:
        iterator = tqdm(iterator, total=n_steps, desc="Simulating", unit="step")  # type: ignore

    for step in iterator:
        t = step * dt_s
        # handle schedule
        mode = schedule[next_idx - 1][1]
        if next_idx < len(schedule) and t >= schedule[next_idx][0]:
            next_idx += 1
            mode = schedule[next_idx - 1][1]

        if mode == "idle":
            # zero current step without control
            meas = twin.batt.step(0.0, dt_s)
            meas["i_A"], meas["v_V"] = 0.0, meas.pop("v_term")
            alerts = twin.ctrl.check_alerts({**meas, "v_term": meas["v_V"]})
            twin._log(t, mode, meas, len(alerts))
        else:
            twin.step(dt_s, mode)

    # Optional CSV logging
    if csv_path:
        try:
            import csv
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                cols = list(twin.history.keys())
                writer.writerow(cols)
                for i in range(len(twin.history["t_s"])):
                    writer.writerow([twin.history[c][i] for c in cols])
        except Exception as e:
            print("CSV write failed:", e)

    if plot:
        try:
            import matplotlib.pyplot as plt
            t = np.array(twin.history["t_s"]) / 3600.0
            plt.figure(); plt.plot(t, twin.history["i_A"]);   plt.xlabel("time (h)"); plt.ylabel("I (A)")
            plt.figure(); plt.plot(t, twin.history["v_V"]);   plt.xlabel("time (h)"); plt.ylabel("V (V)")
            plt.figure(); plt.plot(t, twin.history["temp_C"]);plt.xlabel("time (h)"); plt.ylabel("Temp (°C)")
            plt.figure(); plt.plot(t, twin.history["soc"]);   plt.xlabel("time (h)"); plt.ylabel("SOC")
            plt.tight_layout()
        except Exception as e:
            print("Plotting failed:", e)

    return twin


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EV Charger + Battery Digital Twin")
    parser.add_argument("--sim-seconds", type=float, default=86400, help="Total simulation time in seconds (default: 1 day)")
    parser.add_argument("--sim-hours", type=float, default=None, help="Alternative to --sim-seconds. If set, overrides it.")
    parser.add_argument("--dt", type=float, default=1.0, help="Timestep in seconds")
    parser.add_argument("--no-controller", action="store_true", help="Disable safety/controller logic (open-loop)")
    parser.add_argument("--plot", action="store_true", help="Plot time series at the end")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write CSV log")
    parser.add_argument("--mode", type=str, default="charge", choices=["charge","discharge","idle"], help="Fixed mode when not using schedules")
    parser.add_argument("--enable-discharge", action="store_true", help="Allow discharge mode (V2G)")
    parser.add_argument("--cc", type=float, default=150.0, help="CC current setpoint in A (magnitude)")
    parser.add_argument("--cv", type=float, default=410.0, help="Pack CV voltage in V")
    parser.add_argument("--schedule", type=str, default=None, help="Optional schedule as 't:mode,t:mode' in seconds, e.g. '0:charge,43200:idle,64800:charge'")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar (if installed)")
    args = parser.parse_args()

    # Build params
    bp = BatteryParams()
    cp = ChargerParams(cc_current_A=args.cc, cv_voltage_V=args.cv, enable_discharge=args.enable_discharge)

    # Build schedule
    mode_schedule: Optional[List[Tuple[float,str]]] = None
    total_s = args.sim_seconds if args.sim_hours is None else args.sim_hours*3600
    if args.schedule:
        items = []
        for kv in args.schedule.split(','):
            t_s, m = kv.split(':')
            items.append((float(t_s), m))
        # sort and clamp last to total
        items = sorted(items, key=lambda x: x[0])
        mode_schedule = items
    else:
        # Single-mode schedule for entire sim
        mode_schedule = [(0.0, args.mode)]

    twin = demo(sim_s=total_s, dt_s=args.dt, mode_schedule=mode_schedule,
                plot=args.plot, use_controller=(not args.no_controller), csv_path=args.csv,
                show_progress=(not args.no_progress), bp=bp, cp=cp)

    # Summary print
    last = {k: v[-1] for k, v in twin.history.items() if len(v)}
    alerts_total = int(sum(twin.history["alerts_count"]))
    print({**last, "alerts_total": alerts_total, "controller": not args.no_controller})
