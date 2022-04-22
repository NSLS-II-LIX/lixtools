import copy
import json
import time
import types

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mcr_const.constraints.nist.special_shape import ConstraintWithFunction, ConstraintGuinier
from mcr_const.constraints.nist.special_shape import halfgau_mod_gaussian
from mcr_const.constraints.nist.trilinear import ConstraintGlobalPrefactor, SharedGlobalPrefactor
from pymcr.constraints import ConstraintNorm
from pymcr.mcr import NNLS, McrAR

from lixtools.hdf import h5sol_HPLC


def plot_mcr_concentration_scattering_profile(qgrid, conc, xs, ax_conc, ax_xs, specie_names, step_name="Step 1"):
    colors = sns.color_palette("Dark2", conc.shape[0])
    ax_protein_conc = ax_conc.twinx()
    for i, conc in enumerate(conc):
        if i == 0:
            ls = '--'
            cur_ax = ax_conc
        else:
            ls = '-'
            cur_ax = ax_protein_conc
        cur_ax.plot(conc, lw=1.5, ls=ls, c=colors[i], label=specie_names[i] if i < len(specie_names) else None)
    ax_conc.spines['bottom'].set_visible(False)
    ax_conc.tick_params(labeltop=False)
    ax_conc.set_ylabel("Water")
    ax_conc.legend(loc='upper left')
    ax_conc.set_xlabel("Frame #")
    ax_protein_conc.xaxis.tick_bottom()
    ax_protein_conc.set_xlabel("Frame #")
    ax_protein_conc.spines['top'].set_visible(False)
    ax_protein_conc.legend()
    ax_protein_conc.set_ylabel("Protein")
    title = f"Concentration --- {step_name}"
    ax_conc.set_title(title)

    plt.figure()
    colors = sns.color_palette("Dark2", xs.shape[0])
    for i, spec in enumerate(xs):
        if i == 0:
            ls = '--'
        elif i >= 1 + len(specie_names):
            ls = 'dotted'
        else:
            ls = '-'
        ax_xs.plot(qgrid, spec, ls=ls, lw=1.5, c=colors[i], label=specie_names[i] if i < len(specie_names) else None)
    ax_xs.legend()
    ax_xs.set_xlabel(r"q ($\AA^{-1}$)")
    ax_xs.set_ylabel("Intensity")
    ax_xs.set_yscale("log")
    ax_xs.set_xscale("log")
    title = f"XS --- {step_name}"
    ax_xs.set_title(title)
    ax_xs.set_ylim([xs[1:len(specie_names)].min() * 0.5, xs[1:1 + len(specie_names)].max() * 2])


def create_step1_constraint(peak_pos_guess, max_half_width, max_height, tot_steps, opt_method):
    shared_pf = SharedGlobalPrefactor()

    c_constraints = list()
    c_constraints.append(ConstraintGlobalPrefactor(stage=1, shared_prefactor=shared_pf))
    for i, (pos, hw) in enumerate(zip(peak_pos_guess, max_half_width)):
        ci = (np.r_[0:tot_steps],
              np.full(tot_steps, dtype='int', fill_value=i + 1))
        area, center, width, distortion = hw * 0.5 * max_height, pos, hw, 1.0
        ig = [area, center, width, distortion]
        bd = [(0, area * 2),
              (center - hw * 0.5, center + hw * 0.5),
              (2, hw),
              (-hw * 1.5, hw * 1.5)]
        c_constraints.append(ConstraintWithFunction(
            line_indices=[ci],
            func=halfgau_mod_gaussian,
            initial_guess=ig,
            bounds=bd,
            method=opt_method))
    c_constraints.append(ConstraintNorm())
    c_constraints.append(ConstraintGlobalPrefactor(stage=2, shared_prefactor=shared_pf))
    st_constraints = []
    return c_constraints, st_constraints, shared_pf


def create_step2_constraint(peak_pos_guess, max_half_width, max_height, tot_steps, opt_method, qgrid,
                            scale_factor_on_qgrid, guinier_q_ranges, grad_threshes, default_rg=50.0):
    c_constraints, st_constraints, shared_pf = create_step1_constraint(
        peak_pos_guess, max_half_width, max_height, tot_steps, opt_method)
    nq = qgrid.shape[0]
    for i, (q_guinier, q_max), lgt in zip(range(1, len(peak_pos_guess) + 1),
                                          guinier_q_ranges,
                                          grad_threshes):
        ci = (np.full(nq, dtype='int', fill_value=i),
              np.r_[0:nq])
        st_constraints.append(ConstraintGuinier(line_indices=[ci],
                                                qgrid=qgrid,
                                                qscale_vector=scale_factor_on_qgrid,
                                                q_max=q_max,
                                                q_guinier=q_guinier,
                                                linear_grad_thresh=lgt,
                                                mix_ratio=1.0,
                                                default_rg=default_rg))
    return c_constraints, st_constraints, shared_pf


def generate_step2_xs_guess(step1_unscale_xs, qgrid, q_fit_ranges, grad_threshes, scale_factor_on_qgrid):
    xs_guess = [step1_unscale_xs[0] * scale_factor_on_qgrid]
    for i, (xs, (qs, qe), lgt) in enumerate(zip(step1_unscale_xs[1:],
                                                q_fit_ranges,
                                                grad_threshes)):
        gxs = ConstraintGuinier.fit_guinier_spec(qgrid, xs, qe, 20, linear_grad_thresh=lgt)
        gxs[qgrid > qs] = xs[qgrid > qs]
        xs_guess.append(gxs * scale_factor_on_qgrid)
    xs_guess = np.stack(xs_guess)
    return xs_guess


def subtract_buffer_mcr(dd2s, qgrid, 
                        peak_pos_guess: str, max_half_width: str, iframe_bg: int, guinier_q_ranges: str,
                        grad_threshes: str,
                        opt_methods=('dogbox', 'trf'),
                        scale_exp: float = 3.0, max_height: float = 0.05, max_mcr_q: float = 0.6,
                        out_bound_scale=1.0E-4,
                        mcr_tol_increase_1: float = 2.0, mcr_max_iter_1: int = 20,
                        mcr_tol_increase_2: float = 100.0, mcr_max_iter_2: int = 20,
                        sn=None, ax1_xs=None, ax1_conc=None, ax2_xs=None, ax2_conc=None, debug=False):
    
    peak_pos_guess = json.loads(f'[{peak_pos_guess}]')
    max_half_width = json.loads(f'[{max_half_width}]')
    guinier_q_ranges = json.loads(f'[{guinier_q_ranges}]')
    grad_threshes = json.loads(f'[{grad_threshes}]')
    assert set(opt_methods) < {'dogbox', 'trf', 'lm'}
    step1_opt_method, step2_opt_method = opt_methods
    assert len(max_half_width) == len(peak_pos_guess)
    assert len(guinier_q_ranges) == len(peak_pos_guess)
    assert len(guinier_q_ranges[0]) == 2
    assert len(grad_threshes) == len(peak_pos_guess)

    nf = dd2s.shape[1]
    scale_factor_on_qgrid = qgrid ** scale_exp
    scale_factor_on_qgrid[qgrid > max_mcr_q] *= out_bound_scale
    xs_scaled = dd2s.T * scale_factor_on_qgrid
    specie_names = ['Water'] + \
                   [f"Protein {i}" for i in range(1, len(peak_pos_guess) + 1)]

    t1 = time.time()
    if debug is True:
        print("start processing: performing MCR ...")

    # Step1 MCR, No Guinier Constraint
    step1_xs_guess = xs_scaled[[iframe_bg] + peak_pos_guess]
    step1_c_constraints, step1_st_constraints, step1_shared_pf = create_step1_constraint(peak_pos_guess, max_half_width,
                                                                                         max_height, nf,
                                                                                         step1_opt_method)
    mcrar = McrAR(c_regr=NNLS(), st_regr=NNLS(), tol_increase=mcr_tol_increase_1, max_iter=mcr_max_iter_1,
                  c_constraints=step1_c_constraints, st_constraints=step1_st_constraints)
    mcrar.fit(xs_scaled, ST=step1_xs_guess)
    step1_conc = mcrar.C_opt_.T.copy() / step1_shared_pf.prefactor
    step1_scaled_xs = mcrar.ST_opt_.copy()
    step1_rev_xs = step1_scaled_xs / scale_factor_on_qgrid
    plot_mcr_concentration_scattering_profile(qgrid, step1_conc, step1_rev_xs, ax1_conc, ax1_xs, specie_names,
                                              step_name="Step 1")

    # Step2 MCR, Enforce Guinier Constraint
    step2_xs_guess = generate_step2_xs_guess(step1_rev_xs, qgrid, guinier_q_ranges, grad_threshes,
                                             scale_factor_on_qgrid)
    step2_c_constraints, step2_st_constraints, step2_shared_pf = create_step2_constraint(
        peak_pos_guess, max_half_width, max_height, nf, step2_opt_method, qgrid, scale_factor_on_qgrid,
        guinier_q_ranges, grad_threshes)
    mcrar = McrAR(c_regr=NNLS(), st_regr=NNLS(), tol_increase=mcr_tol_increase_2, max_iter=mcr_max_iter_2,
                  c_constraints=step2_c_constraints, st_constraints=step2_st_constraints)
    mcrar.fit(xs_scaled, ST=step2_xs_guess)
    step2_conc = mcrar.C_opt_.T.copy() / step2_shared_pf.prefactor
    step2_scaled_xs = mcrar.ST_opt_.copy()
    step2_rev_xs = step2_scaled_xs / scale_factor_on_qgrid
    plot_mcr_concentration_scattering_profile(qgrid, step2_conc, step2_rev_xs, ax2_conc, ax2_xs, specie_names,
                                              step_name="Step 2")

    """    
    dd2s = (step2_conc[1:].T @ step2_rev_xs[1:]).T  # leave water out
    self.d1s[sn]['subtracted'] = []
    for i in range(nf):
        d1c = copy.deepcopy(self.d1s[sn]['merged'][i])
        d1c.data = dd2s[:, i]
        self.d1s[sn]['subtracted'].append(d1c)

    self.save_d1s(sn, debug=debug)
    """
    if debug is True:
        t2 = time.time()
        print("done, time lapsed: %.2f sec" % (t2 - t1))

    return step2_conc,step2_rev_xs

def bind_subtract_buffer_mcr(dt: h5sol_HPLC):
    dt.subtract_buffer_MCR = types.MethodType(subtract_buffer_mcr, dt)
    return dt
