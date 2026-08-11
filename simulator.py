"""
simulator.py
============
First-principles digital twin of a non-isothermal, jacket-cooled CSTR
running the irreversible exothermic reaction A -> B (first-order Arrhenius
kinetics). This is the classic Bequette-style benchmark reactor known for
open-loop input multiplicity (it genuinely has multiple steady states near
Tc ~ 300-305K), which makes it a realistic, non-trivial testbed rather than
a toy linear system.

Why simulate instead of downloading a dataset? Real plant data is
proprietary and rarely comes with clean fault labels / onset times, which
is exactly what's needed to train and *evaluate* fault detection and RUL
models. A first-principles simulator with injected, labeled faults is the
standard approach used to build public benchmarks in this space (e.g. the
Tennessee Eastman Process was built the same way). It also means this
project has zero external data dependency, which matters for an unattended
overnight run.

Governing equations (component + energy balance on the reactor):
    dCa/dt = (q/V)(Caf - Ca) - k(T) * Ca
    dT/dt  = (q/V)(Tf - T) + (-dH/(rho*Cp)) * k(T) * Ca - (UA/(V*rho*Cp))*(T - Tc)
    k(T)   = k0 * exp(-E/(R*T))

Tc (jacket temperature) is the manipulated variable, moved by a PID
controller to hold T at a setpoint. A proportional level controller
adjusts outlet flow q to hold reactor level near its setpoint.
"""

import numpy as np
from scipy.integrate import solve_ivp


class PID:
    """Position-form PID controller with output clamping and simple
    (clamp-based) anti-windup."""

    def __init__(self, Kc, tau_I, tau_D, out_min, out_max, dt, bias):
        self.Kc = Kc
        self.tau_I = tau_I
        self.tau_D = tau_D
        self.out_min = out_min
        self.out_max = out_max
        self.dt = dt
        self.bias = bias
        self.integral = 0.0
        self.prev_error = 0.0
        self._first = True

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self._first = True

    def step(self, setpoint, measurement):
        error = setpoint - measurement
        deriv = 0.0 if self._first else (error - self.prev_error) / self.dt
        self._first = False

        trial_integral = self.integral + error * self.dt
        raw = self.bias + self.Kc * (error + trial_integral / self.tau_I + self.tau_D * deriv)

        if raw > self.out_max:
            out = self.out_max
        elif raw < self.out_min:
            out = self.out_min
        else:
            out = raw
            self.integral = trial_integral  # only integrate when not saturated

        self.prev_error = error
        # how hard the controller is "pushing" against its limits, useful
        # as a health/RUL signal (0 = far from saturation, 1 = saturated)
        span = (self.out_max - self.out_min)
        saturation = 0.0
        if span > 0:
            saturation = max(0.0, min(1.0, (self.out_max - out) / span if out <= self.bias
                                       else (out - self.out_min) / span))
        return out, error

    def would_saturate(self, out):
        return out <= self.out_min + 1e-6 or out >= self.out_max - 1e-6


def cstr_derivatives(Ca, T, q, V, Caf, Tf, k0, E_over_R, deltaH, rho, Cp, UA, Tc):
    k = k0 * np.exp(-E_over_R / T)
    dCa = (q / V) * (Caf - Ca) - k * Ca
    dT = (q / V) * (Tf - T) + (-deltaH / (rho * Cp)) * k * Ca - (UA / (V * rho * Cp)) * (T - Tc)
    return dCa, dT


def rk4_step(Ca, T, dt, **kwargs):
    """Integrate the CSTR ODEs over one control interval [0, dt], holding
    all manipulated/exogenous variables constant over the step (standard
    zero-order-hold assumption for digital control simulation).

    This system is numerically stiff (the Arrhenius term is extremely
    sensitive to T), so a naive fixed-step RK4 with dt=1 min diverges in
    a single step. We use scipy's adaptive LSODA solver internally instead
    -- it automatically takes many small sub-steps where needed while
    still only reporting the state at the end of the control interval, so
    the control loop structure (discrete PID acting every dt) is
    unchanged.
    """
    def rhs(t, y):
        return cstr_derivatives(y[0], y[1], **kwargs)

    sol = solve_ivp(rhs, (0.0, dt), y0=[Ca, T], method="LSODA", rtol=1e-6, atol=1e-9)
    if not sol.success:
        # Extremely defensive fallback: if the solver ever fails, hold state
        # constant rather than propagate NaN/inf through the rest of the episode.
        return Ca, T
    Ca_next, T_next = sol.y[0, -1], sol.y[1, -1]
    return float(Ca_next), float(T_next)


class FaultState:
    """Tracks a single fault's evolution over the course of an episode
    and returns the current process/sensor modifiers it implies."""

    def __init__(self, name, kind, onset_min, severity, ramp_min=0.0):
        self.name = name
        self.kind = kind              # None, 'abrupt', or 'incipient'
        self.onset_min = onset_min
        self.severity = severity      # final/target severity (interpretation depends on fault)
        self.ramp_min = ramp_min

    def current_severity(self, t_min):
        if self.kind is None or t_min < self.onset_min:
            return 0.0
        if self.kind == "abrupt":
            return self.severity
        # incipient: linear ramp from 0 to full severity over ramp_min
        frac = min(1.0, (t_min - self.onset_min) / max(self.ramp_min, 1e-6))
        return self.severity * frac

    def is_active(self, t_min):
        return self.kind is not None and t_min >= self.onset_min


def apply_fault_modifiers(fault: FaultState, t_min, base_params):
    """Returns a dict of multiplicative/additive modifiers to apply at
    time t_min, given the fault's current severity."""
    s = fault.current_severity(t_min)
    mods = dict(
        Caf_mult=1.0, Tf_add=0.0, UA_mult=1.0, k0_mult=1.0,
        T_sensor_bias=0.0, Ca_sensor_bias=0.0, valve_stiction_frac=0.0,
    )
    if fault.name == "feed_disturbance":
        mods["Caf_mult"] = 1.0 + s * np.sign(hash(fault.name) % 2 - 0.5 + 1e-9)  # deterministic-ish sign
        mods["Caf_mult"] = 1.0 - s  # simpler & consistent: fault reduces feed strength
    elif fault.name == "feed_temp_disturbance":
        mods["Tf_add"] = s  # s carries actual K offset (see sampling code)
    elif fault.name == "fouling":
        mods["UA_mult"] = 1.0 - s
    elif fault.name == "catalyst_deactivation":
        mods["k0_mult"] = 1.0 - s
    elif fault.name == "sensor_bias_T":
        mods["T_sensor_bias"] = s
    elif fault.name == "sensor_bias_Ca":
        mods["Ca_sensor_bias"] = s
    elif fault.name == "valve_stiction":
        mods["valve_stiction_frac"] = s
    return mods


class CSTRSimulator:
    """Runs one episode of the plant, applying an (optional) fault, and
    records a full multivariate sensor time series plus ground-truth
    labels needed for supervised ML downstream."""

    def __init__(self, cfg):
        self.cfg = cfg["simulator"]
        self.env = cfg["operating_envelope"]

    def run_episode(self, rng, fault: FaultState, T_setpoint=None, q_setpoint=None, Caf_nom=None):
        c = self.cfg
        dt = c["dt_minutes"]
        n_steps = int(c["episode_hours"] * 60 / dt)

        if T_setpoint is None:
            T_setpoint = rng.uniform(*self.env["T_setpoint_range"])
        if q_setpoint is None:
            q_setpoint = rng.uniform(*self.env["q_setpoint_range"])
        if Caf_nom is None:
            Caf_nom = rng.uniform(*self.env["Caf_range"])

        Ca, T = c["Ca_init"], c["T_init"]
        level = c["level_init"]
        Tc_prev_actual = c["Tc_init"]

        pid = PID(c["pid"]["Kc"], c["pid"]["tau_I"], c["pid"]["tau_D"],
                  c["pid"]["Tc_min"], c["pid"]["Tc_max"], dt, bias=c["Tc_init"])
        level_Kc = c["level_control"]["Kc"]

        noise = c["noise"]
        analyzer_period = int(c["analyzer_sample_minutes"] / dt)
        analyzer_delay_steps = int(c["analyzer_delay_minutes"] / dt)

        records = []
        Ca_history = []  # to implement analyzer dead time

        for i in range(n_steps):
            t_min = i * dt
            mods = apply_fault_modifiers(fault, t_min, c)

            Caf_eff = Caf_nom * mods["Caf_mult"]
            Tf_eff = c["Tf"] + mods["Tf_add"]
            UA_eff = c["UA_nominal"] * mods["UA_mult"]
            k0_eff = c["k0"] * mods["k0_mult"]

            # --- control: measured T includes sensor bias + noise ---
            T_meas = T + mods["T_sensor_bias"] + rng.normal(0, noise["T_std"])
            Tc_cmd, _ = pid.step(T_setpoint, T_meas)

            # valve stiction: actual Tc lags/sticks relative to command
            stiction = mods["valve_stiction_frac"]
            Tc_actual = Tc_prev_actual + (1.0 - stiction) * (Tc_cmd - Tc_prev_actual)
            Tc_prev_actual = Tc_actual

            # --- level control sets outlet flow, which sets q used in the CSTR ---
            level_meas = level + rng.normal(0, noise["level_std"])
            q_out = q_setpoint + level_Kc * (level_meas - c["level_init"])
            q_out = float(np.clip(q_out, 50.0, 150.0))
            level += (q_setpoint - q_out) * dt / 10.0  # /10 = cross-sectional-area-like scaling

            # --- integrate reactor ODEs one step ---
            Ca, T = rk4_step(
                Ca, T, dt, q=q_out, V=c["V"], Caf=Caf_eff, Tf=Tf_eff,
                k0=k0_eff, E_over_R=c["E_over_R"], deltaH=c["deltaH"],
                rho=c["rho"], Cp=c["Cp"], UA=UA_eff, Tc=Tc_actual,
            )
            Ca = max(Ca, 0.0)
            Ca_history.append(Ca)

            # --- fast (cheap, frequent) sensor readings, noisy ---
            q_meas = q_out + rng.normal(0, noise["q_std"])
            Tc_meas = Tc_actual + rng.normal(0, noise["Tc_std"])

            # --- slow (expensive) analyzer reading of Ca: infrequent + dead time + own bias/noise ---
            Ca_analyzer = np.nan
            if i % analyzer_period == 0 and i - analyzer_delay_steps >= 0:
                true_delayed_Ca = Ca_history[i - analyzer_delay_steps]
                Ca_analyzer = (true_delayed_Ca + mods["Ca_sensor_bias"]
                               + rng.normal(0, noise["Ca_analyzer_std"]))

            sev = fault.current_severity(t_min)
            records.append(dict(
                t_min=t_min,
                T_meas=T_meas, Tc_meas=Tc_meas, Tc_cmd=Tc_cmd, q_meas=q_meas,
                level_meas=level_meas, T_setpoint=T_setpoint, q_setpoint=q_setpoint,
                Caf_nom=Caf_nom, Ca_analyzer=Ca_analyzer, Ca_true=Ca, T_true=T,
                fault_name=fault.name if fault.kind is not None else "none",
                fault_active=int(fault.is_active(t_min)),
                fault_severity=sev,
                pid_saturated=int(pid.would_saturate(Tc_cmd)),
            ))

        import pandas as pd
        return pd.DataFrame.from_records(records)
