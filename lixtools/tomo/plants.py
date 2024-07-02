from lixtools.hdf import h5xs_scan
from lixtools.hdf.scan import calc_tomo
from lixtools.tomo.common import make_map_from_overall_attr
import numpy as np
import multiprocessing as mp

def calc_CI(dt: type(h5xs_scan), sn='overall', data_key="tomo", 
            cell_key='int_cell_Iq', amor_key='int_amor_Iq',
            sc=1., ref_key="absorption", ref_cutoff=0.02)-> None:
    """ sc is needed to account for the intensity difference between cellulose and amorphous 
        components due to the way they are calculated: span of q-range, shaping factor
    """
    mm = dt.proc_data[sn][data_key][cell_key].copy()
    dc = dt.proc_data[sn][data_key][cell_key].d
    da = dt.proc_data[sn][data_key][amor_key].d*sc
    idx = (dt.proc_data[sn][data_key][ref_key].d>=ref_cutoff)
    mm.d[idx] = (dc-da)[idx]/dc[idx]
    mm.d[~idx] = 0
    mm.d[np.isinf(mm.d)] = 0
    dt.proc_data[sn][data_key]['CI'] = mm
    dt.save_data(save_sns=[sn], save_data_keys=[data_key], save_sub_keys=['CI'])


# see Rüggeberg et al. J Struct Biol 2013, though there appear to be errors in the equations (Appendix)
# but the final expressions for the phi positions are correct (from Entwistle et al. 2007)
# nu is the MFA
# alpha is the rotation along the growth direction
# 2theta is the scattering angle
from scipy.optimize import nnls

def phiC(nu, alpha, theta):
    if np.fabs(nu)<0.03:
        tt = -np.tan(nu)*(np.tan(theta)*np.sin(alpha)-np.cos(alpha))
    else:
        tt = np.tan(nu)**2*((np.tan(theta)*np.sin(alpha))**2-np.cos(alpha)**2)
        tt = np.sqrt(1.-tt)-1
        tt /= (np.tan(theta)*np.sin(alpha)+np.cos(alpha))*np.tan(nu)
        
    return 2.*np.arctan(tt)

def get_MFA_basis_set(phi, prof, w0=3, dmfa=4, q0=1.6, xene=15.0, include_constant=False):
    """ (phi, prof) is the angular profile, it may not be centered on phi=0, for now get center by weight
        decompose the profile into basis vectors that correspond to a set of MFAs, from 0 to 90deg, at step size of dmfa
        data collected at x-ray energy of xene (keV)
    """
    
    wl = 2.*3.1416*1973/(xene*1000)
    theta0 = .5*np.arcsin(q0*wl/(4.*np.pi))

    idx = ((phi>=-90)&(phi<=90))
    phi00 = np.sum(phi[idx]*prof[idx]**4)/np.sum(prof[idx]**4)
    
    phi_comp = {}
    phi_comp0 = {}
    xx = np.linspace(-np.pi, np.pi, 3601)
    p1 = phi-phi00
    mfs = np.arange(0, 91, dmfa)
    for mfa in mfs:
        h,b = np.histogram(np.degrees(phiC(np.radians(mfa), xx, theta0)), bins=np.linspace(-180,180,182))
        p0 = (b[1:]+b[:-1])/2 
        gb = np.exp(-(p0/w0)**2)
        h = np.convolve(h, gb, mode='same')/np.sum(gb)
        h1 = np.interp(p1, p0, h)+np.interp(p1, p0+180, h)+np.interp(p1, p0-180, h)
        phi_comp[mfa] = h1
        phi_comp0[mfa] = h

    if include_constant:
        phi_comp['C'] = np.ones_like(h1) # already subtracted

    AA = np.vstack([phi_comp[k] for k in phi_comp.keys()]).T
    AA0 = np.vstack([phi_comp0[k] for k in phi_comp0.keys()]).T  
    
    return mfs,AA

def prep_mfa_data(dt: type(h5xs_scan), rot_center, algorithm="pml_hybrid", 
                  num_iter=60, ref_key="absorption", ref_cutoff=0.02, base=0)-> None:
    """ base is subtracted as scattering background 
    """
    phi = dt.get_h5_attr("overall/Iphi/merged", "phigrid")
    idx = ((phi>=-90)&(phi<=90))
    phi0 = phi[idx]
    mms = []
    for i in np.arange(len(phi))[idx]:
        mm = make_map_from_overall_attr(dt, dt.proc_data['overall']['Iphi']['subtracted'][:, i], 
                                        template_grp=ref_key, map_name=None)
        mms.append(mm)

    dt.proc_data['overall']['maps']['Iphi'] = mms
    dt.save_data(save_sns='overall', save_data_keys=['maps'], save_sub_keys=['Iphi'])
        
    pool = mp.Pool(8)
    jobs = []
    kwargs = {'center':rot_center, 'algorithm': algorithm, 'num_iter': num_iter}

    print(f"processing {dt.fn.split('/')[-1][:-7]}          ")
    for i,mm in enumerate(mms):
        print(f" map #{i}           \r", end="")
        jobs.append(pool.map_async(calc_tomo, [(i, mm, kwargs)]))

    pool.close()
    results = {}
    for job in jobs:
        i,data = job.get()[0]
        tm = dt.proc_data['overall']['tomo'][ref_key].copy()
        tm.d = data
        results[i] = tm 
        print(f"data received for {i}                \r", end="")
    pool.join() 
    tms = [results[k] for k in sorted(results.keys())]
        
    mm = dt.proc_data['overall']['tomo'][ref_key].copy()
    mm.d = np.nansum([abs(pp)*(tm.d-base) for pp,tm in zip(phi0,tms)], axis=0)
    idx = (mm.d>=ref_cutoff)
    mm.d[~idx] = 0
    mm.d[idx] /= np.nansum([(tm.d-base) for pp,tm in zip(phi0,tms)], axis=0)[idx]
    
    dt.proc_data['overall']['tomo']['Iphi'] = tms
    dt.proc_data['overall']['tomo']['mfa_a'] = mm    
    dt.save_data(save_sns='overall', save_data_keys=['tomo'], save_sub_keys=['Iphi', 'mfa_a'])
    dt.set_h5_attr("overall/maps/Iphi", "phi", phi0)
    dt.set_h5_attr("overall/tomo/Iphi", "phi", phi0)
