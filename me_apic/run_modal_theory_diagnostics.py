"""Modal-order selection, Proposition-1 audit, and modal PSD markers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.signal import welch

from run_baseline import Parameters, assemble_beam
from run_pantograph_experiments import (
    ControllerParameters, PantographParameters, adaptive_burst_width,
    chain_matrix, design_modal_lqr, moving_contact_vector, simulate, target_force,
)


def system():
    beam=Parameters(); p=PantographParameters(); control=ControllerParameters()
    kb,mb,free=assemble_beam(beam); nb=len(free); n=nb+3
    k=np.zeros((n,n));m=np.zeros((n,n));c=np.zeros((n,n))
    k[:nb,:nb]=kb;m[:nb,:nb]=mb
    k[nb:,nb:]=chain_matrix(p.stiffness_N_m)
    m[nb:,nb:]=np.diag(p.masses_kg);c[nb:,nb:]=chain_matrix(p.damping_Ns_m)
    b=moving_contact_vector(.5*beam.span_length,beam,free)
    return beam,p,control,k,m,c,b


def main() -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family":"Times New Roman","font.size":8,
                         "pdf.fonttype":42,"axes.grid":True,"grid.alpha":.25})
    out=Path("revision_results");out.mkdir(exist_ok=True)
    beam,p,control,k,m,c,b=system();kc=p.contact_stiffness_N_m
    w2,phi=eigh(k+kc*np.outer(b,b),m)
    freq=np.sqrt(np.maximum(w2,0))/(2*np.pi)
    bu=phi.T@np.r_[np.zeros(m.shape[0]-1),1.0]
    cf=kc*(b@phi)
    score=np.abs(cf*bu)/np.maximum(np.sqrt(w2),1e-12)
    cumulative=np.cumsum(score)/np.sum(score)
    n95=int(np.searchsorted(cumulative,.95)+1)
    selected=control.modal_modes
    modal_rows=[]
    for i in range(min(40,len(freq))):
        modal_rows.append({"mode":i+1,"frequency_Hz":freq[i],
                           "dynamic_residue":score[i],
                           "cumulative_participation":cumulative[i],
                           "selected":int(i<selected)})
    with (out/"modal_selection.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(modal_rows[0]));w.writeheader();w.writerows(modal_rows)

    shapes,gain,P=design_modal_lqr(m,c,k,b,kc,modes=selected,
                                   force_weight=control.modal_force_weight)
    km=shapes.T@(k+kc*np.outer(b,b))@shapes;cm=shapes.T@c@shapes;n=selected
    a0=np.block([[np.zeros((n,n)),np.eye(n)],[-km,-cm]])
    binp=shapes.T@np.r_[np.zeros(m.shape[0]-1),1.0]
    B=np.r_[np.zeros(n),binp][:,None];a1=a0-B@gain[None,:]
    on_max=float(eigh(a1.T@P+P@a1,P,eigvals_only=True).max())
    off_max=float(eigh(a0.T@P+P@a0,P,eigvals_only=True).max())
    audits=[]; spectra={}
    for speed in (300.,350.):
        run=simulate("ME-APIC",speed,beam,p,control,feedback_kind="modal_lqr")
        lv=float(np.nanmax(np.abs(np.diff(run.lyapunov)/np.diff(run.time))))
        gap=control.modal_energy_floor_force_factor*target_force(speed)**2
        bound=min(max(control.adaptive_burst_min_s,control.modal_energy_deadline_s),
                  control.adaptive_burst_min_s+gap/lv)
        observed=float(np.min(np.diff(run.starts)))
        atr=float(100*np.mean(run.active))
        alpha1=max(0.0,-on_max);alpha0=max(0.0,off_max)
        margin=alpha1*atr/100-alpha0*(1-atr/100)
        audits.append({"speed_kmh":speed,"alpha1_certified_s_1":alpha1,
                       "alpha0_s_1":alpha0,"duty_margin_s_1":margin,
                       "empirical_LV_per_s":lv,"theoretical_dwell_bound_s":bound,
                       "observed_min_interstart_s":observed,
                       "full_ISS_condition_verified":int(margin>0)})
        mask=(run.time>=.05*run.time[-1])&(run.time<=.95*run.time[-1])
        spectra[speed]=welch(run.force[mask]-np.mean(run.force[mask]),
                             fs=1/(run.time[1]-run.time[0]),nperseg=min(512,np.sum(mask)))
    with (out/"proposition_audit.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(audits[0]));w.writeheader();w.writerows(audits)

    fig,axes=plt.subplots(1,2,figsize=(7.2,2.75))
    axes[0].plot(np.arange(1,31),100*cumulative[:30],"o-",ms=3,color="#2E7D32")
    axes[0].axhline(95,color="0.3",ls="--",lw=.8,label="95% criterion")
    axes[0].axvline(selected,color="#984EA3",ls=":",lw=1,label="$N_m=16$")
    axes[0].set(xlabel="Retained modal order",ylabel="Cumulative dynamic residue (%)")
    axes[0].legend(frameon=False,fontsize=7)
    for speed,color in ((300.,"#377EB8"),(350.,"#E41A1C")):
        f,psd=spectra[speed];axes[1].semilogy(f,psd,color=color,label=f"{speed:.0f} km/h")
    for fmode in freq[:selected]: axes[1].axvline(fmode,color="0.55",lw=.35,alpha=.7)
    axes[1].axvspan(0,freq[selected-1],color="#984EA3",alpha=.08,label="retained band")
    axes[1].set(xlim=(0,220),xlabel="Frequency (Hz)",ylabel=r"PSD (N$^2$/Hz)")
    axes[1].legend(frameon=False,fontsize=7)
    fig.tight_layout();fig.savefig(out/"modal_selection_spectrum.pdf",bbox_inches="tight");fig.savefig(out/"modal_selection_spectrum.png",dpi=300,bbox_inches="tight");plt.close(fig)
    print({"n95":n95,"selected":selected,"cum_selected":float(cumulative[selected-1]),
           "selected_band_Hz":float(freq[selected-1]),"on_generalized_max":on_max,
           "off_generalized_max":off_max,"audit":audits})


if __name__ == "__main__": main()

