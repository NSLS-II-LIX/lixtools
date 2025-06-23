import numpy as np
import pylab as plt
from py4xs.hdf import h5xs,h5exp,lsh5,find_field
from py4xs.data2d import MatrixWithCoords
from lixtools.hdf import h5xs_an,h5xs_scan
from py4xs.utils import get_grid_from_bin_ranges
from lixtools.notebooks import display_data_scanning
import os,glob,time,PIL,h5py
import multiprocessing as mp

import scipy,cv2
from scipy.signal import butter,filtfilt,find_peaks
from scipy import ndimage

from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import NMF,MiniBatchNMF
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans

from skimage import filters
from skimage.restoration import rolling_ball
from skimage.registration import phase_cross_correlation as pcc
import io

from skbeam.core.fitting.xrf_model import construct_linear_model

def hop(phi, I, plot=False):
    # first need to find the peak and trim data range to -90 to 90 deg
    # make sure that that phi covers the full azimuthal angle range
    if np.max(phi)-np.min(phi)<360:
        raise Exception(f"phi range too narrow: {np.min(phi), np.max(phi)}")
    pks,_ = find_peaks(I, height=np.max(I)/2, distance=0.4*len(I))    
    for pk in pks:    
        if pk>len(phi)/4:
            break
    phi0 = phi[pk]
    
    idx = tuple([(phi>=phi0-90) & (phi<=phi0+90)])
    phi00 = np.sum(I[idx]*phi[idx])/np.sum(I[idx])
    
    # all angular position should fall within [-90, 90]
    phi1 = phi-phi00
    idx = (phi1<-90)
    idx1 = (phi1>90)
    phi1[idx] -= 180*(np.floor(phi1[idx]/180+0.5))
    phi1[idx1] -= 180*(np.floor(phi1[idx1]/180+0.5))

    if plot:
        plt.figure()
        plt.plot(phi1,I,'.')
    
    phi1 = np.radians(phi1)
    c2b = np.sum(I*np.cos(phi1)**2)/(np.sum(I))
    return (2.*c2b-1)/2

def get_hop_from_map(d, xrange, plot=False):
    phi,I,ee = d.apply_symmetry().line_profile(direction="y", xrange=xrange)
    return hop(phi, I, plot)

def get_roi(d, qphirange):
    return np.nanmean(d.apply_symmetry().roi(*qphirange).d)

def BW_lowpass_filter(data, cutoff, order=4):
    b, a = butter(order, cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def remove_zingers(data, filt=[2, 1.5], thresh=4, thresh_abs=0.1):
    mmd = ndimage.gaussian_filter(data, filt)
    idx = ((data-mmd)>mmd/thresh) & (data>thresh_abs)
    mmd2 = np.copy(data)
    mmd2[idx] = np.nan
    
    return mmd2

def remove_zingers_1d(q, I, q0=0.15, radius=3000):
    """ expect the data to have high intensity at low xx
        large radius value seems necessary
        alternatively, use small radius, but perform rolling ball on log(I)
    """
    I1 = np.copy(I)
    I1[q>q0] = rolling_ball(I[q>q0], radius=radius)
    
    return I1

def fix_absorption_map(dt, sname='overall', map_data_key="maps", ref_trans=1.26):
    """ sometimes the incident beam intensity monitor behaves strangely
        the first few data points may be abnomally low
        assume that the incident intensity should be relatively stable
    """
    mi = dt.proc_data[sname][map_data_key]['incident'].d.copy()
    mt = dt.proc_data[sname][map_data_key]['transmission'].d.copy()
    
    m_std = np.std(mi)
    m_mean = np.mean(mi)
    thresh = np.min([10*m_std, m_mean/3])
    idx = (np.fabs(mi-m_mean)>thresh) 
    mt[idx] = np.nan      

    # in principle this can be determined from the transmission sinogram itself
    # if there is high confidence that there are large areas when beam does not go throug the sample
    # but there are other complications, e.g. beam intensity monitor not reliable
    # the transmission value corresponding to empty beam should be well-known
    #h,b = np.histogram(mt[np.isfinite(mt)], bins=100)
    #vbkg = (b[0]+b[1])/2
    ma = -np.log(mt/ref_trans) 
    ma[ma<0] = 0
    dt.proc_data[sname][map_data_key]['absorption'].d = ma
    
    dt.save_data(save_sns=[sname], save_data_keys=[map_data_key], 
                 save_sub_keys=['absorption'])
    
"""
def scf(q, q0=0.08, ex=2):
    sc = 1./(np.power(q, -ex)+1./q0)
    return sc
"""
def scf(q, q0=0.28284, ex=2):
    """ originally defined as sc = 1./(np.power(q, -ex)+1./q0), with default values q0=0.08, ex=2
        this equivalent to scf(np.power(q0, 1/ex), ex) in the new definition, or q0=0.28284
    """
    sc = 1./(np.power(q, -ex)/np.power(q0, -ex)+1.0)
    return sc    

def create_XRF_link(dfn, fn):
    """ pyXRF looks for data under /xrfmap/detsum/counts
        this function creates a h5 that contains a external link to the XRF data 
        that belong to dfn/sn
    """
    with h5py.File(fn, 'w') as fh5, h5py.File(dfn, 'r', swmr=True) as dfh5:
        fh5.create_group("/xrfmap/detsum")
        sn = list(dfh5.keys())[0]
        gn = find_field(dfh5, "xsp3_image", sn)
        fh5["/xrfmap/detsum/counts"] = h5py.ExternalLink(dfn, f"{sn}/{gn}/data/xsp3_image")

def bin_q_data(args):    
    """
    proc_data is dt.proc_data[sn]['qphi']['merged']
    sc is the shaping factor
    q_range=[0.01, 2.5], phi_range=None, dezinger='1d'
    """
    fn,sn,sc,q_range,phi_range,dezinger,trans_cor,debug = args
    
    if debug is True:
        print(f"processing started: {sn}            \r", end="")
    dm = MatrixWithCoords()
    with h5py.File(fn, "r", swmr=True) as fh5:
        mm = []
        proc_data = fh5[f'{sn}/qphi']
        ref_trans = np.average(fh5[f'{sn}/attrs/transmitted'])   #### this should be given externally
        trans_data = fh5[f'{sn}/attrs/transmitted']/ref_trans
        dm.xc = proc_data.attrs['xc']
        dm.yc = proc_data.attrs['yc']
        for i in range(len(proc_data['merged/d'])):
            if i%500==0 and debug:
                print(f"- {sn}, {i}                \r", end="")
            dm.d = proc_data['merged/d'][i]
            dm = dm.apply_symmetry()
            dm.d = dm.d*sc
            if dezinger=='2d':
                dm.d = remove_zingers(dm.d)
            q,dd,ee = dm.line_profile(direction="x", xrange=q_range, yrange=phi_range)
            if dezinger=='1d':
                dd = remove_zingers_1d(q, dd)
            if trans_cor:
                if trans_data[i]>0:
                    dd /= trans_data[i]
                else:
                    dd *= 0
            mm.append(dd)            
        
    if debug is True:
        print(f"processing completed: {sn}                   \r", end="")    
        
    return [sn,q,mm]


def bin_phi_data(args):    
    """
    proc_data is dt.proc_data[sn]['qphi']['merged']
    sc is the shaping factor
    q_range, bkg_q_range, dezinger=True
    """
    fn,sn,q_range,bkg_q_range,dezinger,trans_cor,debug = args
    
    if debug is True:
        print(f"processing started: {sn}            \r", end="")
    dm = MatrixWithCoords()
    with h5py.File(fn, "r", swmr=True) as fh5:
        mm = []
        proc_data = fh5[f'{sn}/qphi']
        ref_trans = np.average(fh5[f'{sn}/attrs/transmitted'])   #### this should be given externally
        trans_data = fh5[f'{sn}/attrs/transmitted']/ref_trans
        dm.xc = proc_data.attrs['xc']
        dm.yc = proc_data.attrs['yc']
        for i in range(len(proc_data['merged/d'])):
            if i%500==0 and debug:
                print(f"- {sn}, {i}                \r", end="")
            dm.d = proc_data['merged/d'][i]
            dm = dm.apply_symmetry()

            if dezinger:
                dm.d = remove_zingers(dm.d)
            phi,dd,ee = dm.line_profile(direction="y", xrange=q_range, bkg_range=bkg_q_range)

            if trans_cor:
                if trans_data[i]>0:
                    dd /= trans_data[i]
                else:
                    dd *= 0
            mm.append(dd)            
        
    if debug is True:
        print(f"processing completed: {sn}                   \r", end="")    
        
    return [sn,phi,mm]


def get_q_data(dt, q0=0.08, ex=2, q_range=[0.01, 2.5], ext="", phi_range=None, save_to_overall=True,
               dezinger='1d', trans_cor=True, debug=True):
    """ 
    """        
    N = len(dt.samples)
    pool = mp.Pool(N)
    jobs = []
    for sn in dt.samples:
        #qgrid = dt.proc_data[dt.samples[0]]['qphi']['merged'][0].xc    # this wouldn't work if the q-phi maps are already loaded
        qgrid = dt.get_h5_attr(f"{dt.samples[0]}/qphi", "xc")
        #qgrid = dt.qgrid        # should be the same as above
        sc = scf(qgrid, q0=q0, ex=ex)
        job = pool.map_async(bin_q_data, 
                             [(dt.fn,sn,sc,q_range,phi_range,dezinger,trans_cor,debug)])  
        jobs.append(job)

    pool.close()
    ret = {}
    for job in jobs:
        [sn, q, mm] = job.get()[0]
        ret[sn] = mm
        print(f"data received: sn={sn}             \r", end="")
    pool.join()
    print()

    #sn = 'overall'
    nm = f"Iq{ext}"
    if save_to_overall:
        sns = ['overall']
    else:
        sns = dt.samples
    
    for sn in sns:
        if sn=='overall':
            data = np.vstack([ret[_] for _ in sorted(dt.samples)])
        else:
            data = np.array(ret[sn])
        dt.add_proc_data(sn, nm, 'merged', data)
        dt.save_data(save_sns=[sn], save_data_keys=nm, save_sub_keys="merged")
        dt.set_h5_attr(f"{sn}/{nm}/merged", "qgrid", q) 
    
def sub_bkg_q(dt, bkg_x_range=[0.03, 0.08], ext="", bkg_thresh=None, mask=None, save_to_overall=True):
    """ bkg_thresh
    """
    if save_to_overall:
        sns = ['overall']
    else:
        sns = dt.samples
    
    for sn in sns:    
        data = dt.proc_data[sn][f"Iq{ext}"]['merged']
        xcor = dt.get_h5_attr(f"{sn}/Iq{ext}/merged", 'qgrid')
        xidx = ((xcor>bkg_x_range[0]) & (xcor<bkg_x_range[1]))
        if bkg_thresh is None:
            dd1 = np.average(data[:, xidx], axis=1)
            c,b = np.histogram(dd1[~np.isinf(dd1)], bins=100, range=[0, np.mean(dd1[~np.isinf(dd1)])])
            bkg_thresh = b[1:][np.argwhere(c>np.max(c)/2)].flatten()[0]
            #bkg_thresh = np.min(dd1)*1.2

        mmb = []
        mm = []
        for i in range(data.shape[0]):
            dd = data[i][:]
            if np.average(dd[xidx])<bkg_thresh:
                mmb.append(dd)

        if len(mmb)<=1:
            raise Exception(f"did not find valid bkg (thresh={bkg_thresh}) ...")

        bkg = np.average(mmb, axis=0)
        mm = data-bkg
        mm[mm<0] = 0

        idx = (np.sum(mm[:, 200:300], axis=1)<0.01)  # to exlude the data that are missing WAXS patterns
        mm[idx,:] = 0                                # set to zero values, no need to track which ones are excluded

        dt.proc_data[sn][f"Iq{ext}"]['subtracted'] = np.vstack(mm)
        dt.save_data(save_sns=[sn], save_data_keys=f"Iq{ext}", save_sub_keys="subtracted")
        dt.set_h5_attr(f"{sn}/Iq{ext}/subtracted", "qgrid", xcor) 
        dt.set_h5_attr(f"{sn}/Iq{ext}/merged", "bkg", bkg) 
        

def get_phi_data(dt, q_range=[0.01, 2.5], bkg_q_range=None,
                 ext="", bkg_thresh=0.6e-2, save_to_overall=True,
                 dezinger=True, trans_cor=True, debug=False):
    """ 
    """
    N = len(dt.samples)
    pool = mp.Pool(N)
    jobs = []
    for sn in dt.samples:
        job = pool.map_async(bin_phi_data, 
                             [(dt.fn,sn,q_range,bkg_q_range,dezinger,trans_cor,debug)])  
        jobs.append(job)

    pool.close()
    ret = {}
    for job in jobs:
        [sn, phi, mm] = job.get()[0]
        ret[sn] = mm
        print(f"data received: sn={sn}             \r", end="")
    pool.join()
    print()
    
    nm = f"Iphi{ext}"
    if save_to_overall:
        sns = ['overall']
    else:
        sns = sorted(dt.samples)

    for sn in sns:
        if sn=='overall':
            data = np.vstack([ret[_] for _ in sorted(dt.samples)])
        else:
            data = np.array(ret[sn])

        dt.add_proc_data(sn, nm, 'merged', data)
        dt.save_data(save_sns=sn, save_data_keys=nm, save_sub_keys="merged")
        dt.set_h5_attr(f"{sn}/{nm}/merged", "phigrid", phi) 

def sub_bkg_phi(dt, bkg_x_range=[70, 110], ext="", bkg_thresh=None, mask=None, save_to_overall=True):
    """ bkg_thresh
    """
    if save_to_overall:
        sns = ['overall']
    else:
        sns = dt.samples
    
    for sn in sns:    
        data = dt.proc_data[sn][f"Iphi{ext}"]['merged']
        xcor = dt.get_h5_attr(f"{sn}/Iphi{ext}/merged", 'phigrid')
        xidx = ((xcor>bkg_x_range[0]) & (xcor<bkg_x_range[1]))
        if bkg_thresh is None:
            dd1 = np.average(data[:, xidx], axis=1)
            # take non-zero values to avoid problem with missing WAXS frames
            c,b = np.histogram(dd1[dd1>0], bins=100, range=[0, np.mean(dd1[np.isfinite(dd1)])])
            bkg_thresh = b[1:][np.argwhere(c>np.max(c)/2)].flatten()[0]
            #bkg_thresh = np.min(dd1)*1.2

        mmb = []
        mm = []
        for i in range(data.shape[0]):
            dd = data[i][:]
            if np.average(dd[xidx])<bkg_thresh:
                mmb.append(dd)

        if len(mmb)<=1:
            raise Exception(f"did not find valid bkg (thresh={bkg_thresh}) ...")

        bkg = np.average(mmb, axis=0)
        mm = data-bkg
        mm[mm<0] = 0

        dt.proc_data[sn][f"Iphi{ext}"]['subtracted'] = np.vstack(mm)
        dt.save_data(save_sns=[sn], save_data_keys=f"Iphi{ext}", save_sub_keys="subtracted")
        dt.set_h5_attr(f"{sn}/Iphi{ext}/subtracted", "phigrid", xcor) 
        dt.set_h5_attr(f"{sn}/Iphi{ext}/merged", "bkg", bkg) 
        

def estimate_Nc(x, mm, Ni=20, offset=0.1, cutoff=0.01):
    """ use SVD to estimate the number of eigen vectors needed to describe the dataset
        cutoff specifies the relative value of the lowest eigen value to be included (default 1% of the first)
    """
    V,S,U = randomized_svd(mm.T, Ni)
    print("SVD diagonal elements: ", S)
    eig_vectors = V*S
    coefs = U

    N = len(S[S>=cutoff*S[0]])
    
    fig, axs = plt.subplots(1,2,figsize=(9,5), gridspec_kw={'width_ratios': [2, 1]})
    for i in range(N):
        axs[0].plot(x, eig_vectors[:,i]-i*offset)
    axs[1].semilogy(S, "ko")
    axs[1].semilogy(N-1, S[N-1], "r.")
    
    return eig_vectors[:N]


def get_evs(x, mms, N=5, max_iter=5000, offset=0.1, use_minibatch=False, **kwargs):
    """ multiple samples/datasets
        dts and mms should have the same length, mms is contains the background-subtracted data
    """
    if use_minibatch:
        model = MiniBatchNMF(n_components=N, max_iter=max_iter, **kwargs)
    else:
        model = NMF(n_components=N, max_iter=max_iter, **kwargs)
    W = model.fit_transform(np.vstack(mms).T)
    eig_vectors = W
    coefs = model.components_
    N = model.n_components_
    print(f"NMF stopped after {model.n_iter_} iterations, err = {model.reconstruction_err_}")

    plt.figure(figsize=(6,5))
    for i in range(eig_vectors.shape[1]):
        plt.plot(x, eig_vectors[:,i]-i*offset)    
    
    return eig_vectors,coefs,model


def make_map_from_overall_attr(dt, attr, sname="overall", 
                               template_grp="int_saxs", map_name=None, correct_for_transsmission=False):
    #if an not in self.proc_data['overall']['attrs'].keys():
    #    raise Exception(f"attribue {an} cannot be found.")

    sl = 0
    maps = []
    if sname=="overall":
        sns = dt.h5xs.keys()
    else:
        sns= [sname]
        
    for sn in sns:
        #if not 'scan' in dt.attrs[sn].keys():
        #    get_scan_parms(dt.h5xs[sn], sn)
        
        ll = np.prod(dt.attrs[sn]['scan']['shape'])
        m = MatrixWithCoords()
        #m.d = np.copy(dt.proc_data['overall']['attrs'][an][sl:sl+ll].reshape(dt.attrs[sn]['scan']['shape']))
        m.d = np.copy(attr[sl:sl+ll].reshape(dt.attrs[sn]['scan']['shape']))
        sl += ll
        
        m.xc = dt.attrs[sn]['scan']['fast_axis']['pos']
        m.yc = dt.attrs[sn]['scan']['slow_axis']['pos']
        m.xc_label = dt.attrs[sn]['scan']['fast_axis']['motor']
        m.yc_label = dt.attrs[sn]['scan']['slow_axis']['motor']
        if dt.attrs[sn]['scan']['snaking']:
            for i in range(1,dt.attrs[sn]['scan']['shape'][0],2):
                m.d[i] = np.flip(m.d[i])

        maps.append(m)

    # assume the scans are of the same type, therefore start from the same direction
    if len(maps)>0:
        mm = maps[0].merge(maps[1:])
    else:
        mm = maps[0]
    
    nidx = np.isnan(dt.proc_data[sname]['maps'][template_grp].d)
    mm.d[nidx] = np.nan
    
    if correct_for_transsmission:
        td = dt.proc_data[sname]['maps']['transmitted'].d[~nidx]
        avg_t = np.average(td)
        mm.d[~nidx] /= (td/avg_t)
    
    if map_name is not None:
        dt.proc_data[sname]['maps'][map_name] = mm
    else:
        return mm
    
def make_ev_maps(dts, x, eig_vectors, coefs, res=None, name='q', abs_cor=False, template_grp="int_saxs"):    
    sl = 0
    N = eig_vectors.shape[-1]
    maps = [f'ev{i}_{name}' for i in range(N)]+[f'res_{name}']
    
    for i in range(len(dts)):
        dt = dts[i]        
        dt.load_data(read_data_keys=["attrs"], quiet=True)
        
        """
        for sn in dt.samples:
            ll = len(dt.proc_data[sn]['attrs'][template_grp])   #'transmission'
            for j in range(N):
                dt.proc_data[sn]['attrs'][f'ev{j}_{name}'] = coefs[j,sl:sl+ll]
            dt.proc_data[sn]['attrs'][f'res_{name}'] = res[sl:sl+ll]
            sl += ll

        dt.make_map_from_attr(attr_names=[f'ev{i}_{name}' for i in range(N)], correct_for_transsmission=abs_cor)
        dt.make_map_from_attr(attr_names=[f'res_{name}'], correct_for_transsmission=abs_cor)
        """
        
        ll = dt.proc_data['overall']['maps'][template_grp].d.size
        for j in range(N):
            make_map_from_overall_attr(dt, coefs[j,sl:sl+ll], template_grp=template_grp, 
                                       map_name=f'ev{j}_{name}', correct_for_transsmission=abs_cor)
        make_map_from_overall_attr(dt, res[sl:sl+ll], template_grp=template_grp, 
                                   map_name=f'res_{name}', correct_for_transsmission=abs_cor)
        sl += ll
        
        if not 'attrs' in dt.proc_data['overall'].keys():
            dt.proc_data['overall']['attrs'] = {}
        dt.proc_data['overall']['attrs'][f'evs_{name}'] = eig_vectors
        dt.proc_data['overall']['attrs'][f'ev_{name}'] = x
        dt.save_data(save_sns='overall', save_data_keys=['attrs'], save_sub_keys=[f'evs_{name}', f'ev_{name}'], quiet=True)
        
def make_maps_from_Iq(dts, save_overall=True, int_data="subtracted",
                      q_list={"int_cellulose": [1.05, 1.15], "int_amorphous": [1.25, 1.35], "int_saxs": [0.05, 0.2]},
                      template_grp="transmission", abs_cor=False, quiet=True): 
    sks = list(q_list.keys())
    for dt in dts:       
        dt.load_data(read_data_keys=["attrs"], quiet=quiet)
        if save_overall:
            sns = ['overall']
        else:
            sns = dt.samples
        
        for sn in sns:
            attr = {}
            q = dt.get_h5_attr(f"{sn}/Iq/subtracted", 'qgrid')
            Iq = dt.proc_data[sn]['Iq'][int_data]
            for k in sks:
                idx = ((q>=q_list[k][0]) & (q<=q_list[k][1]))
                attr[k] = np.average(Iq[:,idx], axis=1)  # averaging instead of summing to facilitate comparison between different attributes
                if not save_overall:
                    dt.add_proc_data(sn, 'attrs', k, attr[k])

        if save_overall:
            for k in sks:
                make_map_from_overall_attr(dt, attr[k], template_grp=template_grp, 
                                           map_name=k, correct_for_transsmission=abs_cor)
            dt.save_data(save_data_keys="maps", save_sub_keys=sks, quiet=True)
        else:
            dt.make_map_from_attr(save_overall=False, attr_names=sks, correct_for_transsmission=False)
                    
            
def scale_transmitted_maps(dt, sc=170000, quiet=True):
    if 'transmitted' not in dt.proc_data['overall']['maps'].keys():
        raise Exception(f"transmitted map missing for {dt.fn} ...")
    avg = np.average(dt.proc_data['overall']['maps']['transmitted'].d)
    if avg<0.1*sc or avg>10*sc:
        print(f"either the transmitted map (average value {avg:.2g}) has been scaled already, or the scale factor {sc} is bad.")
    else:
        dt.proc_data['overall']['maps']['transmitted'].d /= sc
        dt.save_data(save_sns='overall', save_data_keys=['maps'], save_sub_keys=['transmitted'], quiet=quiet)
            
def check_ev_tomos(dt, ev_tag, ref_tomo='absorption', quiet=True):
    """ scale the magnitude based on the values from the sinogram
        shift the tomo based on the ref_tomo
        these are necessary when the tomograms are reconstructed using different algorithms
    """
    grp = dt.proc_data['overall']
    tm0 = grp['tomo'][ref_tomo].d
    keys = [k for k in grp['tomo'].keys() if k[:2]=='ev' and ev_tag in k]
    
    for k in keys:
        tm1 = grp['tomo'][k].d*np.nansum(grp['maps'][k].d)/np.nansum(grp['tomo'][k].d)
        tm1[np.isnan(tm1)] = 0
        
        shift = pcc(tm0, tm1)[0]
        grp['tomo'][k].d = scipy.ndimage.shift(tm1, shift)
        
    dt.save_data(save_sns='overall', save_data_keys=['tomo'], save_sub_keys=keys, quiet=quiet)
    
def recombine(coef, method="nmf"):
    """ coef should have the same dimension as dt.proc_data['overall']['attrs'][f'evs_{method}'].shape[1]
    """
    N = dt.proc_data['overall']['attrs'][f'evs_{method}'].shape[1]
    if len(coef)!=N:
        raise Exception(f"shape mismatch: {len(coef)} != {N}")
        
    return np.sum(coef*dt.proc_data['overall']['attrs'][f'evs_{method}'], axis=1)


def get_roi(d, qphirange):
    return np.nanmean(d.apply_symmetry().roi(*qphirange).d)
    
import xraydb,json
from scipy.optimize import nnls

def nnls_decomp(data, basis_set, n_start, n_end, sigma=3):
    """ basis_set x coeff = data
        return coeff
        
        this is typically for XRF elemental analysis
        data is typically 3-dimensional, e.g. (m, k, n) for a k-element detector, rather than (m, n)
        when this function is called from extract_attr(), the last index is enumerated
        therefore the data shape is (m, 1/k), the result shape is (-1, 1/k)
        additionally, work on a limited range of columns (:, n_start:n_end)
    """
    data1 = data[:, n_start:n_end]
    shp = data1.shape
    result = []
    for spec in data1.reshape(-1, shp[-1]):
        try:
            coeff,err = nnls(basis_set.T, spec)
        except: # sometimes NNLS fails, it always doesn't work 100% with filtered data
            coeff,err = nnls(basis_set.T, ndimage.gaussian_filter(spec,sigma))            
        result.append(coeff)
    return np.array(result).reshape(*shp[:-1], -1)

def prep_data_from_pyxrf(dt, ele_list, save_overall=True, xrf_fns=None):
    """ this function populates attrs/xrf_XX based on the results from pyxrf fits
        these results are expected in files listed in h5fns
        when xrf_fns is None, construct the file list assuming the filename is simply appended _xrf after the sample/run name
    """
    if xrf_fns is None:
        xrf_fns = [f"{ssn}_xrf.h5" for ssn in dt.samples]
    if len(xrf_fns) != len(dt.samples):
        raise Exception("incorrect length of {xrf_fns}")
    xrf_list = [f"xrf_{en}" for en in ele_list]

    for xrf_fn,ssn in zip(xrf_fns, dt.samples):
        with h5py.File(xrf_fn, "r") as fh5:
            xrf_elist = list(fh5["xrfmap/detsum/xrf_fit_name"][...])    # list of bytes, not strings
            pyxrf_data = fh5["xrfmap/detsum/xrf_fit"][...]        
        for ele in ele_list:
            i = xrf_elist.index(ele.encode("utf-8"))
            dt.add_proc_data(ssn, 'attrs', f'xrf_{ele}', pyxrf_data[i,:,0])
    dt.save_data(save_data_keys=['attrs'], save_sub_keys=xrf_list)
    dt.make_map_from_attr(save_overall=save_overall, attr_names=xrf_list, correct_for_transsmission=False)            

def prep_XRF_data(dt, ele_list, save_overall=True, pyxrf_param_fn=None,
                  eNstart=310, eNend=1100, eBin=0.01, pk_width=0.167/2.355):
    """ ele_list, e.g.  ['K_K', 'Ca_K', 'Mn_K', 'Fe_K', 'Cu_K', 'Zn_K']
        pyxrf_param_fn: use this if there is a json file from fitting the summed sepctrum using pyxrf 
                        in this case ele_list could include "compton"
        eNstart,eNend specify the range of data to be used from the MCA, with bin size of eBin keV
        3.1keV is right after the Ar peak
    """
    xrf_e_range = np.arange(eNstart, eNend)*eBin

    fluo_line_profiles = {}

    if pyxrf_param_fn: # get the basis spectra from the json file
        with open(pyxrf_param_fn, 'r') as fh:
            params = json.load(fh)
        ch_num = np.arange(eNstart, eNend, dtype=int)
        e_select, matv, e_area = construct_linear_model(ch_num, params, ele_list)
        """
        for ele in ele_list:
            i = e_select.index(ele)
            d = matv[:,i]
            fluo_line_profiles[i] = d/np.max(d)
        """
        for i,ele in enumerate(e_select):
            d = matv[:,i]
            fluo_line_profiles[ele] = d/np.max(d)
        xrf_list = [f"xrf_{en.split('_')[0]}" for en in e_select]
    else: # construct the basis spectra based on the values from xraydb 
        for ele in ele_list:
            e,l = ele.split("_")
            xls = xraydb.xray_lines(e, l)
            spec = np.zeros_like(xrf_e_range)
            for xl in xls.values():
                spec += np.exp(-(xrf_e_range-xl.energy/1000)**2/(2.*pk_width**2))*xl.intensity
            fluo_line_profiles[e] = spec
        xrf_list = [f"xrf_{en.split('_')[0]}" for en in ele_list]
    
    AA = np.vstack(list(fluo_line_profiles.values()))

    print(f"processing {dt.fn.split('/')[-1][:-7]}          ")
    
    for sn in dt.samples:
        print(f" processing {sn}           \r", end="")
        dt.extract_attr(sn, xrf_list, nnls_decomp, "xsp3", "xsp3_image", from_raw_data=True, 
                        basis_set=AA, n_start=eNstart, n_end=eNend)    
    if save_overall:
        sn = 'overall'
    else:
        sn = dt.samples[0]
    dt.add_proc_data(sn, 'XRF', 'basis', AA)
    dt.save_data(save_sns=[sn], save_data_keys=['XRF'], save_sub_keys=['basis'])
    dt.set_h5_attr(f"{sn}/XRF/basis", "ele_list", json.dumps(ele_list))
    dt.save_data(save_data_keys=['attrs'], save_sub_keys=xrf_list)
    dt.make_map_from_attr(save_overall=save_overall, attr_names=xrf_list, correct_for_transsmission=False)
    
def crop(dd, cx, cy, roi_w=0, roi_h=0):
    if roi_h>0:
        y1 = int(cy-roi_h/2)
        y2 = int(cy+roi_h/2)
    else:
        y1 = 0
        y2 = -1
    if roi_w>0:
        x1 = int(cx-roi_w/2)
        x2 = int(cx+roi_w/2)
    else:
        x1 = 0
        x2 = -1
    return dd[y1:y2, x1:x2]


def get_cap_mask(dt, ref_key="xrf_elastic", vmin=60, n_blur=5, n_rep_dilate=3, plot_data=False):
    """ this is useful for measurements done in glass capillaries
        the elastic peak from the XRF data can be used to identify the voxels that correspond to the capillary
        returns the mask for the capillary
        if plot_data is True, plot the comparison between before and after removal of capillary voxels
    """
    kernel = np.ones((n_blur,n_blur),np.uint8)
    data = dt.proc_data['overall']['tomo'][ref_key]
    if isinstance(data, list):
        mm = np.sum([d.d for d in data], axis=0)
    else:
        mm = data.d
    mask = np.array((mm>vmin), dtype=np.float64)
    mask1 = cv2.GaussianBlur(mask, (n_blur,n_blur),0)
    mask2 = cv2.dilate(mask1, kernel, iterations = n_rep_dilate)
    th = filters.threshold_otsu(mask2)

    if plot_data:
        plt.figure(figsize=(8,4))
        plt.subplot(121)
        plt.imshow(mm)
        plt.subplot(122)
        mm[mask2>th] = 0
        plt.imshow(mm)
    
    return (mask2>th)
    

def plot_data(dt: type(h5xs_scan), data_key, sub_keys=None, auto_scale=False, max_w=10, max_h=4, Nx=0, 
              default_cmap='binary', cust_cmaps={}, cmax={}, cmin={}, cm_scale=1., 
              space=0, aspect='auto', sample_in_row=True, roi_w=0, roi_h=0,
              use_alpha_for_all=False, alpha_key="int_cell_Iq", alpha_cutoff=0.03, flat_alpha=False, save_fn=None, dpi=300):
    """
        cust_cmaps a dictionary that specifies sub_keys that should use customized colormaps
    """
    if isinstance(dt, list):
        data0 = [t.proc_data['overall'][data_key] for t in dt]
    else:
        data0 = [dt.proc_data[sn][data_key] for sn in dt.samples]
    
    if sub_keys is None:
        sub_keys = dt.proc_data[sn][data_key].keys()
    data = []
    for dd0 in data0:
        dd = {} 

        for k in sub_keys:
            if isinstance(dd0[k], list):
                mm = dd0[k][0].copy()
                mm.d = np.nansum([m0.d for m0 in dd0[k]], axis=0)
                dd[k] = mm
            else:
                dd[k] = dd0[k]
        dd['alpha'] = dd0[alpha_key].copy()
        if flat_alpha:
            img = dd['alpha'].d
            img = np.array(img/np.nanmax(img)*255, dtype=np.uint8)
            blur = cv2.GaussianBlur(img,(5,5),0)
            ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            _,msk,_,_ = cv2.floodFill(th3, None, [0,0], 127)
            dd['alpha'].d = 1.-np.array((msk==127), dtype=float)
            dd['alpha'].d[dd0[alpha_key].d<alpha_cutoff] = 0
        else:
            dd['alpha'].d -= alpha_cutoff
            dd['alpha'].d[dd['alpha'].d<0] = 0
            dd['alpha'].d /= (0.5*np.nanmax(dd['alpha'].d))
            dd['alpha'].d[dd['alpha'].d>1] = 1
            dd['alpha'].d = np.sqrt(dd['alpha'].d)
        data.append(dd)
        
    mm = data[0][sub_keys[0]]
    asp0 = np.fabs((mm.yc[-1]-mm.yc[0])/(mm.xc[-1]-mm.xc[0]))
    asp = np.fabs((mm.yc[1]-mm.yc[0])/(mm.xc[1]-mm.xc[0]))
    
    cmax0 = {}
    cmin0 = {}
    for k in sub_keys:
        dd = np.hstack([d0[k].d.flatten() for d0 in data])
        xx = np.nanmax(dd)
        xm = np.nanstd(dd)
        cmax0[k] = np.min([xx, xm*8])
        cmin0[k] = 0
    for k in cmax.keys():
        if k in cmax0.keys():
            cmax0[k] = cmax[k]
    for k in cmin.keys():
        if k in cmin0.keys():
            cmin0[k] = cmin[k]
    
    if sample_in_row:
        Nr = len(data)
        Nc = len(sub_keys)
    else:
        Nc = len(data)
        Nr = len(sub_keys)

    if auto_scale:
        fw = np.min([max_w, max_h/Nr/asp0*Nc])
        fh = np.min([max_h, max_w/Nc*asp0*Nr])
    else:
        fw = max_w
        fh = max_h
                    
    fig,axs = plt.subplots(nrows=Nr, ncols=Nc, figsize=(fw,fh))
    if Nr==1:
        axs = np.expand_dims(axs, axis=0)
    elif Nc==1:
        axs = np.expand_dims(axs, axis=1)
    
    for i,d0 in enumerate(data):

        cy,cx = ndimage.center_of_mass(d0['alpha'].d)
        for j,k in enumerate(sub_keys):
            
            if sample_in_row:
                ax = axs[i][j]
            else:
                ax = axs[j][i]

            mask = 1
            if data_key=='tomo':
                mask = d0['alpha'].d
                aspect = 'equal'
            else:
                aspect = aspect
            mm = d0[k].copy()
            Nx0 = len(mm.xc)
            if Nx>Nx0 and data_key=="tomo":
                pw = int((Nx-Nx0)/2)
                mm.d = np.pad(mm.d, pw)
                mask = np.pad(mask, pw)
                #ma.d = np.pad(ma.d, pw)
                #mask = (ma.d>alpha_cutoff)

            cmap = default_cmap
            if k in cust_cmaps.keys():
                cmap = cust_cmaps[k]
            if (k in ["mfa_a"]) | use_alpha_for_all:  # tomo only
                img = crop(mm.d, cx, cy, roi_w=roi_w, roi_h=roi_h)
                ax.imshow(img, aspect=aspect, alpha=crop(mask, cx, cy, roi_w=roi_w, roi_h=roi_h), #np.sqrt(mask), 
                          vmin=cmin0[k]*cm_scale, vmax=cmax0[k]*cm_scale, cmap=cmap) #, cmap="binary")
            else:
                # if flat_alpha, everything masked has zero value
                img = crop(mm.d*mask, cx, cy, roi_w=roi_w, roi_h=roi_h)
                ax.imshow(img, aspect=aspect, 
                          vmin=cmin0[k]*cm_scale, vmax=cmax0[k]*cm_scale, cmap=cmap)
            ax.set_axis_off()
    
    plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, wspace=space, hspace=space)
    if save_fn is not None:
        plt.savefig(f"{save_fn}.png", dpi=dpi)


def make_video(mms, fn, labels=None, fps=3, figsize=(4,4), **kwargs):
    """ mms is a list of MatrixWithCorrds
        fn is the file to write the video to
        labels are displayed as the plot tilte in each frame
        kwargs are passed onto MatrixWithCorrds.plot()
    """
    current_backend = plt.get_backend()
    plt.close('all')
    plt.switch_backend('agg')
    
    if labels is None:
        labels = ["" for _ in range(len(mms))]

    if len(labels)!=len(mms):
        raise Exception(f"mms and labels need to have the sane length: {len(mms)} vs {len(labels)} ...")
    nfr = len(mms)
    
    vidwriter = None    
    frames = []
    for i in range(nfr):
        fig,ax = plt.subplots(figsize=figsize)
        mms[i].plot(ax=ax, **kwargs)
        plt.subplots_adjust(top=0.85, right=0.85)
        ax.set_title(labels[i], y=1.1)
        #plt.show()
        
        # this may be slower
        buf = io.BytesIO()
        fig.savefig(buf, dpi=200)
        buf.seek(0)
        img = PIL.Image.open(buf)
        
        # this produces poorer resolution
        #img = PIL.Image.fromarray(np.array(fig.canvas.buffer_rgba()))

        frames.append(img.copy())
        plt.close()

    plt.switch_backend(current_backend)

    height,width,layers = np.array(frames[0]).shape
    vidwriter = cv2.VideoWriter(f"{fn}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height)) 
    for img in frames:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        vidwriter.write(frame)
    cv2.destroyAllWindows()
    vidwriter.release()    