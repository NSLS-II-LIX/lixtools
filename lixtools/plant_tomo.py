"""
This file provide functions specific to tomography measurements on plants
Assume that the growth direction is roughly aligned with the rotation axis
"""
import numpy as np
import pylab as plt
from py4xs.hdf import h5xs,h5exp,lsh5
from py4xs.data2d import MatrixWithCoords
from lixtools.hdf import h5xs_an,h5xs_scan
from lixtools.hdf.scan import gen_scan_report,calc_tomo
from py4xs.utils import get_grid_from_bin_ranges
from lixtools.notebooks import display_data_scanning
import os,glob,time,PIL

import scipy
from scipy.signal import butter,filtfilt
from scipy.ndimage import gaussian_filter

from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import NMF
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans

from skimage.registration import phase_cross_correlation as pcc

q_bin_ranges = [[[0.0045, 0.0995], 95], [[0.0975, 0.6025], 101], [[0.605, 2.905], 230]]
qgrid0 = get_grid_from_bin_ranges(q_bin_ranges)

def BW_lowpass_filter(data, cutoff, order=4):
    b, a = butter(order, cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def remove_zingers(data, filt=[2, 1.5], thresh=4, thresh_abs=0.1):
    mmd = gaussian_filter(data, filt)
    idx = ((data-mmd)>mmd/thresh) & (data>thresh_abs)
    mmd2 = np.copy(data)
    mmd2[idx] = np.nan
    
    return mmd2

def scf(q, q0=0.08, ex=2):
    sc = 1./(np.power(q, -ex)+1./q0)
    return sc


def plot_data(data, q, skip=237, ax=None, logScale=True):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    for i in range(0,data.shape[0],skip):
        ax.plot(q, data[i,:])
    
    if logScale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    
def get_q_data(dt, q0=0.08, ex=2, q_range=[0.01, 2.5], dezinger=True):
    """ 
    """        
    mm = []
    for sn in dt.samples:
        for i in range(len(dt.proc_data[sn]['qphi']['merged'])):
            dm = dt.proc_data[sn]['qphi']['merged'][i].apply_symmetry()
            dm.d = dm.d*scf(dm.xc, q0=q0, ex=ex)
            if dezinger:
                dm.d = remove_zingers(dm.d)
            q,dd,ee = dm.line_profile(direction="x", xrange=q_range)
            mm.append(dd)
        # temporary fix to pad zeros for the missing SAXS/merged data
        for i in range(len(dt.proc_data[sn]['attrs']['transmitted'])-len(dt.proc_data[sn]['qphi']['merged'])):
            mm.append(np.zeros_like(dd))

    sn = 'overall'
    if not "Iq" in dt.proc_data[sn].keys():
        dt.proc_data[sn]["Iq"] = {}
    dt.proc_data[sn]["Iq"]['merged'] = np.vstack(mm)
    dt.save_data(save_sns=sn, save_data_keys="Iq", save_sub_keys="merged")
    dt.set_h5_attr("overall/Iq/merged", "qgrid", q) 

    
def sub_bkg_q(dt, bkg_x_range=[0.03, 0.08], bkg_thresh=None, mask=None):
    """ bkg_thresh
    """
    data = dt.proc_data['overall']["Iq"]['merged']
    xcor = dt.get_h5_attr("overall/Iq/merged", 'qgrid')
    xidx = ((xcor>bkg_x_range[0]) & (xcor<bkg_x_range[1]))
    if bkg_thresh is None:
        dd1 = np.average(data[:, xidx], axis=1)
        c,b = np.histogram(dd1, bins=100, range=[0, np.mean(dd1)])
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
    
    dt.proc_data['overall']["Iq"]['subtracted'] = np.vstack(mm)
    dt.save_data(save_sns='overall', save_data_keys="Iq", save_sub_keys="subtracted")
    dt.set_h5_attr("overall/Iq/subtracted", "qgrid", xcor) 
    dt.set_h5_attr("overall/Iq/merged", "bkg", bkg) 
        
def get_phi_data(dt, phi_range=[0.01, 2.5], bkg_thresh=0.6e-2, dezinger=True):
    """ 
    """
    mm = []
    mmb = []
    for sn in dt.samples:
        for i in range(len(dt.proc_data[sn]['qphi']['merged'])):
            dm = dt.proc_data[sn]['qphi']['merged'][i].apply_symmetry()
            if dezinger:
                dm.d = remove_zingers(dm.d)
            phi,dd,ee = dm.line_profile(direction="y", xrange=phi_range)
            mm.append(dd)
            if np.max(dd)<bkg_thresh:
                mmb.append(dd)
    
    sn = 'overall'
    if not "Iphi" in dt.proc_data[sn].keys():
        dt.proc_data[sn]["Iphi"] = {}
    dt.proc_data[sn]["Iphi"]['merged'] = np.vstack(mm)
    dt.save_data(save_sns=sn, save_data_keys="Iphi", save_sub_keys="merged")
    dt.set_h5_attr("overall/Iphi/merged", "phigrid", phi) 

def sub_bkg_phi(dt, bkg_x_range=[70, 110], bkg_thresh=None, mask=None):
    """ bkg_thresh
    """
    data = dt.proc_data['overall']["Iphi"]['merged']
    xcor = dt.get_h5_attr("overall/Iphi/merged", 'phigrid')
    xidx = ((xcor>bkg_x_range[0]) & (xcor<bkg_x_range[1]))
    if bkg_thresh is None:
        dd1 = np.average(data[:, xidx], axis=1)
        c,b = np.histogram(dd1, bins=100, range=[0, np.mean(dd1)])
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

    dt.proc_data['overall']["Iphi"]['subtracted'] = np.vstack(mm)
    dt.save_data(save_sns='overall', save_data_keys="Iphi", save_sub_keys="subtracted")
    dt.set_h5_attr("overall/Iphi/subtracted", "phigrid", xcor) 
    dt.set_h5_attr("overall/Iphi/merged", "bkg", bkg) 
        

def estimate_Nc(x, mm, N=20, offset=0.1):
    V,S,U = randomized_svd(mm.T, N)
    print("SVD diagonal elements: ", S)
    eig_vectors = V*S
    coefs = U

    fig, axs = plt.subplots(1,2,figsize=(9,5), gridspec_kw={'width_ratios': [2, 1]})
    for i in range(eig_vectors.shape[1]):
        axs[0].plot(x, eig_vectors[:,i]-i*offset)
    axs[1].semilogy(S, "ko")
    
    return eig_vectors


def get_evs(x, mms, N=5, max_iter=5000, offset=0.1):
    """ multiple samples/datasets
        dts and mms should have the same length, mms is contains the background-subtracted data
    """
    model = NMF(n_components=N, max_iter=max_iter)
    W = model.fit_transform(np.vstack(mms).T)
    eig_vectors = W
    coefs = model.components_
    N = model.n_components_
    print(f"NMF stopped after {model.n_iter_} iterations, err = {model.reconstruction_err_}")

    plt.figure(figsize=(6,5))
    for i in range(eig_vectors.shape[1]):
        plt.plot(x, eig_vectors[:,i]-i*offset)    
    
    return eig_vectors,coefs


def make_map_from_overall_attr(dt, attr, template_grp="int_saxs", map_name=None, correct_for_transsmission=False):
    #if an not in self.proc_data['overall']['attrs'].keys():
    #    raise Exception(f"attribue {an} cannot be found.")

    sl = 0
    maps = []
    for sn in dt.h5xs.keys():
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
    mm = maps[0].merge(maps[1:])
    #if "overall" not in self.proc_data.keys():
    #    self.proc_data['overall'] = {}
    #    self.proc_data['overall']['maps'] = {}
    #self.proc_data['overall']['maps'][an] = mm
    
    nidx = np.isnan(dt.proc_data['overall']['maps'][template_grp].d)
    mm.d[nidx] = np.nan
    
    if correct_for_transsmission:
        td = dt.proc_data['overall']['maps']['transmitted'].d[~nidx]
        avg_t = np.average(td)
        mm.d[~nidx] /= (td/avg_t)
    
    if map_name is not None:
        dt.proc_data['overall']['maps'][map_name] = mm
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
        
def make_maps_from_Iq(dts, 
                      q_list={"int_cellulose": [1.05, 1.15], "int_amorphous": [1.25, 1.35], "int_saxs": [0.05, 0.2]},
                      template_grp="transmission", abs_cor=False, quiet=True): 

    for dt in dts:       
        dt.load_data(read_data_keys=["attrs"], quiet=quiet)
        q = dt.get_h5_attr("overall/Iq/subtracted", 'qgrid')
        Iq = dt.proc_data['overall']['Iq']['subtracted']
        for k in q_list.keys():
            idx = ((q>=q_list[k][0]) & (q<=q_list[k][1]))
            attr = np.sum(Iq[:,idx], axis=1)
            make_map_from_overall_attr(dt, attr, template_grp=template_grp, 
                                       map_name=k, correct_for_transsmission=abs_cor)
            
def scale_transmitted_maps(dt, sc=170000, quiet=True):
    if 'transmitted' not in dt.proc_data['overall']['maps'].keys():
        raise Exception(f"transmitted map missing for {dt.fn} ...")
    avg = np.average(dt.proc_data['overall']['maps']['transmitted'].d)
    if avg<0.1*sc or avg>10*sc:
        print(f"either the transmitted map (average value {avg:.2g}) has been scaled already, or the scale factor {sc} is bad.")
    else:
        dt.proc_data['overall']['maps']['transmitted'].d /= sc
        dt.save_data(save_sns='overall', save_data_keys=['maps'], save_sub_keys=['transmitted'], quiet=quiet)
            
def check_ev_tomos(dt, ref_tomo='absorption', quiet=True):
    """ scale the magnitude based on the values from the sinogram
        shift the tomo based on the ref_tomo
        these are necessary when the tomograms are reconstructed using different algorithms
    """
    grp = dt.proc_data['overall']
    tm0 = grp['tomo'][ref_tomo].d
    keys = [k for k in grp['tomo'].keys() if 'ev' in k]
    
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


def proc_maps(sample_name, qgrid=qgrid0, detectors=None, Nphi=61, 
              attr_list = {"int_cellulose": [1.05, 1.15, -180, 180],
                           "int_amorphous": [1.25, 1.35, -180, 180],
                           "int_saxs":      [0.05, 0.2,  -180, 180]}, 
              raw_data_pattern = "%s-??.h5", replace_path={},
              datakey="qphi", subkey="merged", 
              reprocess=False, abs_cor=False):
    """ 
        assume that the first pass of data processing is already done: qphi maps already exists
    """
    data_file = f"{sample_name}_an2.h5"
    if os.path.exists(data_file):
        dt = h5xs_scan(data_file, load_raw_data=True, replace_path=replace_path)
        dt.load_data()
    else:
        if detectors is None:
            raise Exception("analysis file does not existing yet, must specify detectors ...")
        raw_data_files = sorted(glob.glob(raw_data_pattern % sample_name))
        if len(raw_data_files)==0:
            raise Exception(f"must specify a list of raw data files for {sample_name} ...")
        dt = h5xs_scan(data_file, [detectors, qgrid], Nphi=Nphi, pre_proc="2D")
        dt.import_raw_data(raw_data_files, force_uniform_steps=True, prec=0.001, debug=True)
        
        t0 = time.time()
        dt.process(N=8, debug=True)
        print(f"time elapsed: {time.time()-t0:.1f} sec")
        dt.save_data()

    # build attribute maps
    if (not "overall" in dt.proc_data.keys()) or reprocess:
        if len(attr_list)==0:
            raise Exception("attribute list is empty ...")
            
        fast_axis = dt.attrs[dt.samples[-1]]['scan']['fast_axis']['motor']
        exp = dt.attrs[dt.samples[-1]]['header']['pilatus']['exposure_time']
        for sn,dh5 in dt.h5xs.items():
            dt.get_mon(sn=sn, trigger=fast_axis, exp=exp, force_synch="auto")
            for attr in attr_list.keys():
                dt.extract_attr(sn, attr, get_roi, "qphi", "merged", qphirange=attr_list[attr])
        
        maps_list = list(attr_list.keys())
        dt.make_map_from_attr(attr_names=["absorption"]+maps_list, correct_for_transsmission=abs_cor)
        dt.save_data(save_data_keys=["attrs", "maps"])
        
        imputer = KNNImputer(n_neighbors=3, weights="uniform")
        for attr in maps_list:
            mm = dt.proc_data['overall']['maps'][attr]
            if len(np.where(np.isnan(mm.d)))>0:
                mm.d = imputer.fit_transform(mm.d)
        dt.calc_tomo_from_map(attr_names=maps_list, algorithm="pml_hybrid", num_iter=100)
        dt.save_data(save_sns="overall", save_data_keys=["tomo"])

    return dt