# Battery Parameters

- `capacity_Ah`: Total usable ampere-hours at beginning of life; converted to coulombs via `capacity_As()` for SOC integration and equivalent full-cycle counting.
- `n_series`: Number of cells in series; multiplies cell OCV/limits to get pack-level voltages and establishes the nominal pack voltage.
- `n_parallel`: Parallel strings count; primarily informational because `capacity_Ah` already reflects the full pack, but relevant if scaling to different modules.
- `r0_ohm`: Instantaneous cell-pack ohmic resistance used in the terminal voltage drop `V = OCV - I*r0`, dominating high-frequency IR sag.
- `r1_ohm`: Polarization resistance feeding the first-order RC block; couples with `c1_f` to model transient relaxation (`v1` state in `BatteryModel.step`).
- `c1_f`: Polarization capacitance for the RC element; integrates current to simulate slow diffusion effects that shape voltage recovery.
- `v_min_cell`: Minimum per-cell safety voltage; multiplied by `n_series` to clamp terminal voltage and trigger undervoltage alerts.
- `v_max_cell`: Maximum per-cell safety voltage; multiplied by `n_series` to define pack CV ceiling and overvoltage protection.
- `mass_kg`: Lumped battery+casing mass used in the thermal equation `dT = (q_in - q_out)/(mass * heat_capacity)`.
- `heat_capacity_j_per_kgK`: Specific heat used with `mass_kg` to determine temperature rise per joule of net heat in the lumped thermal model.
- `h_w_per_K`: Effective cooling conductance (`h*A`); subtracts `h*(T-Tamb)` from the heat balance, modeling coolant/ambient heat rejection.
- `t_ambient_C`: Ambient/coolant temperature baseline; the thermal model and derating thresholds reference this starting point.
- `entropic_w_per_A`: Entropic heat coefficient in watts per amp; added to `I^2*R` heating to capture charge/discharge entropy effects (sign-sensitive).
- `cal_k_ref_per_s`: Calendar-aging rate constant at reference temperature; used in `_update_aging` with Arrhenius scaling to decay capacity over time.
- `cyc_k_ref`: Cycle-aging rate constant; combined with C-rate exponent and Arrhenius term to estimate fade per ampere throughput.
- `alpha_cyc`: Exponent on instantaneous C-rate in the cycling fade term, controlling how aggressively high currents age the pack.
- `arrhenius_Ea_cal_J`: Activation energy for calendar fade; governs temperature sensitivity via the Arrhenius factor in `_arrhenius`.
- `arrhenius_Ea_cyc_J`: Activation energy for cycling fade; mirrors the calendar term but applied to throughput-induced degradation.
- `t_warn_C`: Warning threshold for pack temperature; when exceeded, the alert system logs a “WARN: High temperature” message.
- `t_hot_C`: “Hot” threshold triggering overtemperature shutdown alerts and controller derating to zero current.
- `t_crit_C`: Critical runaway temperature referenced in the sigmoid risk score (`thermal_runaway_risk`), shaping risk escalation.
- `derate_start_C`: Temperature at which the controller linearly reduces allowable current (`derate_factor`).
- `derate_stop_C`: Temperature at which allowable current reaches zero, forcing idle until the pack cools.
- `max_charge_power_W`: Upper bound on charging power; combined with terminal voltage to cap commanded current.
- `max_discharge_power_W`: Limit on discharge/regeneration power, used symmetrically when V2G or discharge mode is enabled.
- `max_c_rate`: Maximum C-rate allowed; converts to an amp limit (`capacity_Ah * max_c_rate`) before applying power and thermal checks.

# Charger Parameters

- `cc_current_A`: Constant-current setpoint magnitude; the controller clamps to this value (after applying power, C-rate, and derating limits).
- `cv_voltage_V`: Constant-voltage ceiling for the pack; once terminal voltage meets this value, the controller tapers current to hold it.
- `enable_discharge`: Flag that allows positive current commands (discharge/V2G); when false, controller and open-loop modes only permit charging.
