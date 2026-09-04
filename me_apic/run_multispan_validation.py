"""Independent five-span two-wire/dropper validation for the APIC controllers.

The model is deliberately separate from the single-span development plant.  It
contains Euler--Bernoulli contact and messenger wires, support/steady-arm
springs, and prestressed-linearized tensile droppers.  Parameters follow the
published multi-span set used by Liu et al., Applied Sciences 13 (2023) 6819.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from run_baseline import Parameters, element_matrices, hermite
from run_pantograph_experiments import (
    ControllerParameters, PantographParameters, adaptive_burst_width,
    chain_matrix, design_modal_lqr, target_force, wire_irregularity,
)


@dataclass(frozen=True)
class MultiSpanParameters:
    spans: int = 5
    span_length_m: float = 60.0
    elements_per_span: int = 12
    contact_EI_Nm2: float = 195.0
    contact_tension_N: float = 2.0e4
    contact_density_kg_m: float = 1.35
    messenger_EI_Nm2: float = 131.7
    messenger_tension_N: float = 1.6e4
    messenger_density_kg_m: float = 1.07
    dropper_stiffness_N_m: float = 1.0e5
    messenger_support_stiffness_N_m: float = 5.0e4
    steady_arm_stiffness_N_m: float = 300.0
    rayleigh_alpha_s_1: float = 0.0125
    rayleigh_beta_s: float = 1.0e-4
    dt_s: float = 8.2082e-4


def wire_matrices(elements: int, length: float, density: float,
                  tension: float, bending: float) -> tuple[np.ndarray, np.ndarray]:
    p = Parameters(span_length=length, elements=elements, linear_density=density,
                   axial_tension=tension, bending_stiffness=bending)
    nd = 2*(elements+1)
    k = np.zeros((nd, nd)); m = np.zeros_like(k)
    ke, me = element_matrices(length/elements, p)
    for e in range(elements):
        dof = np.array([2*e, 2*e+1, 2*e+2, 2*e+3])
        k[np.ix_(dof, dof)] += ke
        m[np.ix_(dof, dof)] += me
    return k, m


def assemble_model(mp: MultiSpanParameters, pp: PantographParameters):
    ne = mp.spans*mp.elements_per_span
    length = mp.spans*mp.span_length_m
    kc, mc = wire_matrices(ne, length, mp.contact_density_kg_m,
                           mp.contact_tension_N, mp.contact_EI_Nm2)
    km, mm = wire_matrices(ne, length, mp.messenger_density_kg_m,
                           mp.messenger_tension_N, mp.messenger_EI_Nm2)
    nw = kc.shape[0]
    n = 2*nw+3
    k = np.zeros((n, n)); m = np.zeros_like(k)
    k[:nw, :nw] = kc; m[:nw, :nw] = mc
    k[nw:2*nw, nw:2*nw] = km; m[nw:2*nw, nw:2*nw] = mm
    k[-3:, -3:] = chain_matrix(pp.stiffness_N_m)
    m[-3:, -3:] = np.diag(pp.masses_kg)

    support_nodes = [s*mp.elements_per_span for s in range(mp.spans+1)]
    for node in support_nodes:
        k[2*node, 2*node] += mp.steady_arm_stiffness_N_m
        k[nw+2*node, nw+2*node] += mp.messenger_support_stiffness_N_m

    nominal_positions = (5.0, 10.5, 17.0, 23.5, 30.0, 36.5, 43.0, 49.5, 55.0)
    dropper_nodes: set[int] = set()
    dx = length/ne
    for span in range(mp.spans):
        for local in nominal_positions:
            dropper_nodes.add(int(round((span*mp.span_length_m+local)/dx)))
    for node in sorted(dropper_nodes):
        ic, im = 2*node, nw+2*node
        kd = mp.dropper_stiffness_N_m
        k[ic, ic] += kd; k[im, im] += kd
        k[ic, im] -= kd; k[im, ic] -= kd

    c = mp.rayleigh_alpha_s_1*m + mp.rayleigh_beta_s*k
    c[-3:, -3:] += chain_matrix(pp.damping_Ns_m)
    return k, m, c, nw, ne, length, len(dropper_nodes)


def contact_vector(position: float, ne: int, length: float, nw: int) -> np.ndarray:
    le = length/ne
    e = min(int(np.floor(position/le)), ne-1)
    shape, _ = hermite(position-e*le, le)
    b = np.zeros(2*nw+3)
    b[np.array([2*e, 2*e+1, 2*e+2, 2*e+3])] = -shape
    b[-3] = 1.0
    return b


def rank1_solve(factor, b: np.ndarray, stiffness: float, rhs: np.ndarray) -> np.ndarray:
    y = cho_solve(factor, rhs, check_finite=False)
    z = cho_solve(factor, b, check_finite=False)
    return y-z*(stiffness*(b@y)/(1.0+stiffness*(b@z)))


def simulate(method: str, speed_kmh: float, mp: MultiSpanParameters,
             pp: PantographParameters, cp: ControllerParameters):
    k, m, c, nw, ne, length, ndrop = assemble_model(mp, pp)
    speed = speed_kmh/3.6; dt = mp.dt_s; ref = target_force(speed_kmh)
    load0 = np.zeros(k.shape[0]); load0[-1] = ref
    kfac = cho_factor(k, check_finite=False)
    bmid = contact_vector(2.5*mp.span_length_m, ne, length, nw)
    shapes, gain, riccati = design_modal_lqr(
        m, c, k, bmid, pp.contact_stiffness_N_m,
        modes=cp.modal_modes, force_weight=cp.modal_force_weight,
    )
    b0 = contact_vector(0.0, ne, length, nw)
    r0 = wire_irregularity(0.0)
    q = rank1_solve(kfac, b0, pp.contact_stiffness_N_m,
                    load0+pp.contact_stiffness_N_m*b0*r0)
    qd = np.zeros_like(q); qdd = np.zeros_like(q)
    previous_eq = q.copy()
    horizon = length/speed; time = np.arange(int(np.floor(horizon/dt))+1)*dt
    force = np.zeros_like(time); actuator = np.zeros_like(time)
    actuator_velocity = np.zeros_like(time); active = np.zeros(time.size, dtype=bool)
    force[0] = max(0.0, pp.contact_stiffness_N_m*(b0@q-r0))
    tau = adaptive_burst_width(speed_kmh, cp)
    active_until = tau if method in ("E-APIC", "ME-APIC") else -np.inf
    last_start = 0.0; starts = [0.0] if method != "Passive" else []
    off_force_energy = (force[0]-ref)**2; off_modal_energy = 0.0
    a0 = 1.0/(0.275*dt*dt); a1 = 0.55/(0.275*dt)
    dynfac = cho_factor(k+a0*m+a1*c, check_finite=False)
    lyapunov = np.zeros_like(time)

    for j, t in enumerate(time[:-1]):
        x = speed*t; b = contact_vector(x, ne, length, nw); irr = wire_irregularity(x)
        eq = rank1_solve(kfac, b, pp.contact_stiffness_N_m,
                         load0+pp.contact_stiffness_N_m*b*irr)
        eqd = (eq-previous_eq)/dt if j else np.zeros_like(q)
        eta = shapes.T@m@(q-eq); etad = shapes.T@m@(qd-eqd)
        xm = np.r_[eta, etad]; lyapunov[j] = xm@riccati@xm
        if method == "Passive": on = False
        elif method == "Continuous": on = True
        elif method == "E-APIC":
            if t < active_until: on = True
            else:
                if active[j-1] if j else True: off_force_energy = (force[j]-ref)**2
                trigger = (force[j]-ref)**2 >= cp.event_sigma*off_force_energy+cp.event_floor_N2
                if trigger or t-last_start >= cp.event_deadline_s:
                    starts.append(float(t)); last_start=t; active_until=t+cp.burst_width_s; on=True
                else: on=False
        elif method == "ME-APIC":
            if t < active_until: on = True
            else:
                if active[j-1] if j else True: off_modal_energy = lyapunov[j]
                threshold = cp.modal_energy_sigma*off_modal_energy+cp.modal_energy_floor_force_factor*ref**2
                if lyapunov[j] >= threshold or t-last_start >= cp.modal_energy_deadline_s:
                    starts.append(float(t)); last_start=t; active_until=t+tau; on=True
                else: on=False
        else: raise ValueError(method)
        u = float(np.clip(-gain@xm, -cp.limit_N, cp.limit_N)) if on else 0.0
        active[j] = on; actuator[j] = u
        load = load0.copy(); load[-1] += u
        rhs = load + m@(a0*q+qd/(0.275*dt)+(1/(2*0.275)-1)*qdd)
        rhs += c@(a1*q+(0.55/0.275-1)*qd+dt*(0.55/(2*0.275)-1)*qdd)
        xn = speed*time[j+1]; bn = contact_vector(xn, ne, length, nw); irn = wire_irregularity(xn)
        qn = rank1_solve(dynfac, bn, pp.contact_stiffness_N_m,
                          rhs+pp.contact_stiffness_N_m*bn*irn)
        qddn = (qn-q-dt*qd)/(0.275*dt*dt)-(1/(2*0.275)-1)*qdd
        qdn = qd+dt*((1-0.55)*qdd+0.55*qddn)
        fc = pp.contact_stiffness_N_m*(bn@qn-irn)
        if fc < 0.0:
            qn = cho_solve(dynfac, rhs, check_finite=False)
            qddn = (qn-q-dt*qd)/(0.275*dt*dt)-(1/(2*0.275)-1)*qdd
            qdn = qd+dt*((1-0.55)*qdd+0.55*qddn); fc = 0.0
        q, qd, qdd = qn, qdn, qddn
        force[j+1] = fc; actuator_velocity[j+1] = qd[-1]; previous_eq = eq
    active[-1]=active[-2]; actuator[-1]=actuator[-2]; lyapunov[-1]=lyapunov[-2]
    position = speed*time
    mask = (position >= mp.span_length_m) & (position <= (mp.spans-1)*mp.span_length_m)
    row = {
        "speed_kmh": speed_kmh, "controller": method,
        "std_force_N": float(np.std(force[mask])),
        "mean_force_N": float(np.mean(force[mask])),
        "contact_loss_percent": float(100*np.mean(force[mask] <= 0)),
        "ATR_percent": float(100*np.mean(active[mask])),
        "effort_N2s": float(np.sum(actuator[mask]**2)*dt),
        "mechanical_work_J": float(np.sum(np.abs(actuator[mask]*actuator_velocity[mask]))*dt),
        "events": len(starts),
    }
    return row, (time, position, force, active), ndrop


def main() -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family":"Times New Roman", "font.size":8,
                         "pdf.fonttype":42, "axes.grid":True, "grid.alpha":.25})
    out = Path("multispan_results"); out.mkdir(parents=True, exist_ok=True)
    mp=MultiSpanParameters(); pp=PantographParameters(); cp=ControllerParameters()
    rows=[]; histories={}; ndrop=0
    for speed in (300.0,350.0):
        for method in ("Passive","Continuous","E-APIC","ME-APIC"):
            row,hist,ndrop=simulate(method,speed,mp,pp,cp); rows.append(row); histories[(speed,method)]=hist
            print(speed,method,row,flush=True)
    for speed in (300.0,350.0):
        passive=next(r for r in rows if r["speed_kmh"]==speed and r["controller"]=="Passive")
        for row in rows:
            if row["speed_kmh"]==speed:
                row["improvement_percent"]=100*(passive["std_force_N"]-row["std_force_N"])/passive["std_force_N"]
    with (out/"multispan_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,1,figsize=(7.2,4.5),sharex=False)
    colors={"Passive":"0.45","Continuous":"#377EB8","E-APIC":"#E69F00","ME-APIC":"#009E73"}
    for ax,speed in zip(axes,(300.0,350.0)):
        for method in colors:
            t,x,f,a=histories[(speed,method)]; mask=(x>=120)&(x<=180)
            ax.plot(x[mask],f[mask],lw=.75,color=colors[method],label=method)
        ax.set_ylabel(r"$F_c$ (N)");ax.set_title(f"Central span, {speed:.0f} km/h")
    axes[-1].set_xlabel("Position (m)"); axes[0].legend(ncol=4,frameon=False)
    fig.tight_layout();fig.savefig(out/"multispan_validation.pdf",bbox_inches="tight");fig.savefig(out/"multispan_validation.png",dpi=300,bbox_inches="tight");plt.close(fig)
    (out/"multispan_metadata.json").write_text(json.dumps({"model":asdict(mp),"pantograph":asdict(pp),"controller":asdict(cp),"droppers":ndrop},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()

