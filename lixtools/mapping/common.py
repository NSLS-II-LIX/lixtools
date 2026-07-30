import numpy as np
import pylab as plt
from py4xs.hdf import h5xs,h5exp,lsh5,find_field
from py4xs.data2d import MatrixWithCoords
from lixtools.hdf import h5xs_an,h5xs_scan
from lixtools.notebooks import display_data_scanning
import os,glob,time,PIL,h5py
import multiprocessing as mp

import dask.array as da
from dask.diagnostics import ProgressBar

import scipy,cv2
from scipy.signal import butter,filtfilt,find_peaks
from scipy import ndimage

from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans

from skimage import filters
from skimage.restoration import rolling_ball
from skimage.registration import phase_cross_correlation as pcc
import io

from skbeam.core.fitting.xrf_model import construct_linear_model

def hop(phi, I, plot=False):
    """
        2D implementation of Herman's orientation parameter
        first need to find the peak and trim data range to -90 to 90 deg
        make sure that that phi covers the full azimuthal angle range
    """
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
    """ expect the data to have high intensity at low q
        large radius value seems necessary
        alternatively, use small radius, but perform rolling ball on log(I)
    """
    I1 = np.copy(I)
    I1[q>q0] = rolling_ball(I[q>q0], radius=radius)
    
    return I1
    
def remove_zingers_min(data, q, q0=0.75):
    """ for the specific case of sharp peaks on powder rings
        at each q value, the lowest intensity should correspond to the powder patter
        leave data below q0 intact
    """
    mmd = np.copy(data)
    for i in range(len(q)):
        if q[i]>q0:
            mmd[:, i] = np.ones_like(data[:, i])*np.nanmin(data[:, i])
    mmd[np.isnan(data)] = np.nan

    return mmd

def fix_maps(dt, sname='overall', map_data_key="maps", map_sub_keys=None,
             fix_inf=True, fix_neg=False):

    if not (fix_inf or fix_neg):
        return
    
    if map_sub_keys is None:
        map_sub_keys = list(dt.proc_data[sname][map_data_key].keys())
    elif isinstance(map_sub_keys, str):
        map_sub_keys = [map_sub_keys]
    elif not isinstance(map_sub_keys, list):
        raise Exception(f"Don't know how to handle map_sub_keys: {map_sub_keys}")
    
    for k in map_sub_keys:
        maps = dt.proc_data[sname][map_data_key][k]
        if not isinstance(maps, list):
            maps = [maps]
        for m in maps:
            mask = np.zeros_like(m.d, dtype=np.uint8)
            if fix_inf:
                mask[np.isinf(m.d)] = 1
            if fix_neg:
                mask[m.d<0] = 1

            m.d = cv2.inpaint(np.array(m.d, dtype=np.float32), mask, 3, cv2.INPAINT_TELEA)

    dt.save_data(sname, [map_data_key], map_sub_keys)
        

def fix_absorption_map(dt, sname='overall', map_data_key="maps", ref_trans=None, inpaint=True):
    """ sometimes the incident beam intensity monitor behaves strangely
        the first few data points may be abnomally low
        assume that the incident intensity should be relatively stable
    """
    mi0 = dt.proc_data[sname][map_data_key]['incident'].d.copy()
    mt0 = dt.proc_data[sname][map_data_key]['transmitted'].d.copy()

    # sometimes the the incident and transmitted data contain zero values
    # fill with inpaint
    mi = cv2.inpaint(np.array(mi0, dtype=np.float32), np.array(mi0==0, dtype=np.uint8), 3, cv2.INPAINT_TELEA)
    mt = cv2.inpaint(np.array(mt0, dtype=np.float32), np.array(mt0==0, dtype=np.uint8), 3, cv2.INPAINT_TELEA)
    
    # this will exclude some data points where the intensity may not be reliable
    m_std = np.std(mi)
    m_mean = np.mean(mi)
    #thresh = np.min([10*m_std, m_mean/3])
    thresh = np.min([np.sqrt(m_mean)*10, m_mean/3])
    idx = (np.fabs(mi-m_mean)>thresh) 
    mt[idx] = np.nan      
    
    # in principle this can be determined from the transmission sinogram itself
    # if there is high confidence that there are large areas when beam does not go through the sample
    # but there are other complications, e.g. beam intensity monitor not reliable
    # the transmission value corresponding to empty beam should be well-known
    mt /= mi
    if ref_trans is None:
        h,b = np.histogram(mt[np.isfinite(mt)], bins=100)
        ref_trans = b[-1] #(b[-1]+b[-2])/2
    
    ma = -np.log(mt/ref_trans) 
    ma[np.isinf(ma)] = np.nan
    if inpaint:
        ma = cv2.inpaint(np.array(ma, dtype=np.float32), np.array(np.isnan(ma), dtype=np.uint8), 3, cv2.INPAINT_TELEA)
    
    # min absorption value should be 0 
    h,b = np.histogram(ma[np.isfinite(ma)], bins=100, range=[np.nanmin(ma), np.nanmean(ma)/10])
    ni = np.argmax(h)
    ma -= (b[ni]+b[ni+1])
    ma[ma<0] = 0

    dt.proc_data[sname][map_data_key]['absorption'].d = ma
    
    dt.save_data(save_sns=[sname], save_data_keys=[map_data_key], 
                 save_sub_keys=['absorption', 'incident', 'transmitted'])
    
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
        # pyXRF reads the incident X-ray enengy, in keV
        hdr = json.loads(dfh5[sn].attrs['start'])
        grp = fh5.create_group("/xrfmap/scan_metadata")
        grp.attrs['instrument_mono_incident_energy'] = 1e-3*hdr['energy']['energy']
        grp.attrs['scan_id'] = hdr['scan_id']

def bin_q_data(args):    
    """
    proc_data is dt.proc_data[sn]['qphi']['merged']
    sc is the shaping factor
    q_range=[0.01, 2.5], phi_range=None, dezinger='1d'

    need some way to pass aguments for dezinger    
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
            if dezinger=='min':
                dm.d = remove_zingers_min(dm.d, dm.xc)
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
    """ calculate 1D I(q) intensity profiles form the merged q-phi intensity maps
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
    """ background subtraction for I(q) intensity profiles
        assume the profiles with lowest intensity with bkg_x_range to be background scattering
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
        
def sub_bkg_qphi(dt, bkg_x_range=[0.03, 0.08], ref_trans=5000, bkg_thresh=None):
    """ apaply central symmtry to merged q-phi intensity maps
        find the frames with lowest scattering intensity in bkg_x_range, assume them as background scattering
        perform background subtraction based on transmitted beam intensity
        for consistency between samples, normalized transmitted intensity value to ref_trans
    """
    sns = dt.samples
    with h5py.File(dt.fn, "r+") as fh5:
        xcor = fh5[f"{sns[0]}/qphi"].attrs['xc']
        xidx = ((xcor>bkg_x_range[0]) & (xcor<bkg_x_range[1]))
        ycor = fh5[f"{sns[0]}/qphi"].attrs['yc']
        d0 = fh5[f"{sns[0]}/qphi/merged/d"][0,:,:]
        Np = int(len(ycor)/2)
        mask0 = np.isnan(d0)
        mask1 = np.vstack([mask0[Np:,:], mask0[:Np,:]])
        mask_r = (mask1 & ~mask0)    # fill in the gap
        mask_d = (~mask1 & ~mask0)   # redundant
        
        for sn in sns:    
            print(f"begin processing {sn} ...       \r", end="")
            data = fh5[f"{sn}/qphi/merged/d"][...]
            trans_data = fh5[f'{sn}/attrs/transmitted'][:data.shape[0]]/ref_trans   # issue with transmitted data being too long

            # interpolate zero transmitted beam intensity
            idx_z = (trans_data==0)
            trans_data[idx_z]= np.interp(trans_data[idx_z], trans_data[~idx_z], trans_data[~idx_z])
            
            tt = np.hstack([data[:,Np:,:], data[:,:Np,:]])
            tt[:,mask_r] = data[:,mask_r]            
            tt[:,mask_d] = (tt[:,mask_d]+data[:,mask_d])/2

            tt /= trans_data[:,None,None]
            
            dd1 = np.nanmean(tt[:,:,xidx], axis=(1,2))
            if bkg_thresh is None:
                c,b = np.histogram(dd1[~np.isinf(dd1)], bins=100, range=[0, np.mean(dd1[~np.isinf(dd1)])])
                bkg_thresh = b[1:][np.argwhere(c>np.max(c)/2)].flatten()[0]
    
            bidx = (dd1<bkg_thresh)
            if len(dd1[bidx])<=1:
                raise Exception(f"did not find valid bkg (thresh={bkg_thresh}) ...")
    
            bkg = np.average(tt[bidx,:,:], axis=0)
            tt -= bkg
    
            if "subtracted" in fh5[f"{sn}/qphi"].keys():
                fh5[f"{sn}/qphi/subtracted/d"][...] = tt
            else:
                fh5.create_dataset(f"{sn}/qphi/subtracted/d", data=tt, chunks=True, compression='lzf')
            for k in fh5[f"{sn}/qphi/merged"].attrs.keys():
                fh5[f"{sn}/qphi/subtracted"].attrs[k] = fh5[f"{sn}/qphi/merged"].attrs[k]

def bin_q_data_dask(fn, sn, sc, dezinger, trans_cor, ref_trans, qphi_grp="merged"):    
    """    
    sc is the shaping factor
    q_range=[0.01, 2.5], phi_range=None, dezinger='1d'
    """
    
    with h5py.File(fn, "r", swmr=True) as fh5:
        Iqphi = da.array(fh5[f'{sn}/qphi/{qphi_grp}/d'])
        #ref_trans = np.average(fh5[f'{sn}/attrs/transmitted'])   #### this should be given externally
        trans_data = fh5[f'{sn}/attrs/transmitted'][:Iqphi.shape[0]]/ref_trans    # issue with transmitted data being too long

        if qphi_grp=="merged":
            dm = fh5[f'{sn}/qphi/{qphi_grp}/d'][0]
            Np = int(dm.shape[0]/2)
            w0 = np.zeros_like(dm, dtype=np.int8)
            w0[~np.isnan(dm)] = 1
            w = w0 + np.vstack([w0[Np:,:], w0[:Np,:]])
            Iqphi1 = da.nansum(da.array([Iqphi, da.concatenate([Iqphi[:,Np:,:], Iqphi[:,:Np,:]], axis=1)]), axis=0) / w
        else:
            Iqphi1 = Iqphi

        if dezinger=="median":  # this should be able to get rid of sharp peaks
            Iq = da.nanmedian(Iqphi1, axis=-2)
        else:
            Iq = da.nanmean(Iqphi1, axis=-2)
        
        if dezinger=="1d":  # apply some additional smoothing/peak removal opes
            print("1d dezinger for bin_q_data_dask() not yet implemented")

        if qphi_grp=="merged":  # ignoring trans_cor
            idx = (trans_data>0)
            Iq[idx,:] = (Iq[idx,:].T/trans_data[idx]).T  

        with ProgressBar():
            ret = Iq.compute()
                
    return ret

def get_q_data_dask(dt, q0=0.08, ex=2, q_range=[0.01, 2.5], ext="", save_to_overall=True,
               dezinger=None, trans_cor=True, ref_trans=5000, qphi_grp="merged", debug=True):
    """ 
    """        
    with h5py.File(dt.fn, "r") as fh5:
        qgrid = fh5[f"{dt.samples[0]}/qphi"].attrs['xc']
    idx = ((qgrid>=q_range[0])&(qgrid<=q_range[1]))
    sc = scf(qgrid, q0=q0, ex=ex)
    Iqs = {sn:bin_q_data_dask(dt.fn, sn, sc, dezinger, trans_cor, ref_trans, qphi_grp) for sn in dt.samples}
    
    nm = f"Iq{ext}"
    if save_to_overall:
        sns = ['overall']
    else:
        sns = dt.samples
    
    for sn in sns:
        if sn=='overall':
            data = da.concatenate([Iqs[_][:,idx] for _ in sorted(dt.samples)])
        else:
            data = Iqs[sn][:,idx]
        #print("saving data for ", sn)
        #with ProgressBar():
        #    data.to_hdf5(dt.fn, f"{sn}/{nm}/merged")
        #dt.set_h5_attr(f"{sn}/{nm}", "type", "ndarray")        
        dt.add_proc_data(sn, nm, qphi_grp, data)
        dt.save_data(sn, nm, qphi_grp)
        dt.set_h5_attr(f"{sn}/{nm}/{qphi_grp}", "qgrid", qgrid[idx]) 

def sub_bkg_q_dask(dt, bkg_x_range=[0.03, 0.08], ext="", bkg_thresh=None, mask=None, save_to_overall=True):
    """ bkg_thresh
    """
    if save_to_overall:
        sns = ['overall']
    else:
        sns = dt.samples
    
    with h5py.File(dt.fn, "r+") as fh5:
        xcor = fh5[f"{sns[0]}/Iq{ext}/merged"].attrs['qgrid']
        xidx = ((xcor>bkg_x_range[0]) & (xcor<bkg_x_range[1]))
        for sn in sns:    
            data = da.array(fh5[f"{sn}/Iq{ext}/merged"])
            dd1 = da.average(data[:, xidx], axis=1).compute()
            if bkg_thresh is None:
                c,b = np.histogram(dd1[~np.isinf(dd1)], bins=100, range=[0, np.mean(dd1[~np.isinf(dd1)])])
                bkg_thresh = b[1:][np.argwhere(c>np.max(c)/2)].flatten()[0]
    
            bidx = (dd1<bkg_thresh)
            if len(dd1[bidx])<=1:
                raise Exception(f"did not find valid bkg (thresh={bkg_thresh}) ...")
    
            bkg = da.average(data[bidx,:], axis=0).compute()
            data -= bkg
    
            if "subtracted" in fh5[f"{sn}/Iq{ext}"].keys():
                fh5[f"{sn}/Iq{ext}/subtracted"][...] = data.compute()
            else:
                fh5.create_dataset(f"{sn}/Iq{ext}/subtracted", data=data.compute())
            #data.to_hdf5(dt.fn, f"{sn}/Iq{ext}/subtracted")
            fh5[f"{sn}/Iq{ext}/subtracted"].attrs["qgrid"] = xcor 
            fh5[f"{sn}/Iq{ext}/merged"].attrs["bkg"] = bkg 

def bin_phi_data_dask(fn,sn,q_range,dezinger,trans_cor,ref_trans,qphi_grp):    
    """
    q_range, bkg_q_range, dezinger=True
    """
    
    with h5py.File(fn, "r", swmr=True) as fh5:
        Iqphi = da.array(fh5[f'{sn}/qphi/{qphi_grp}/d'])
        #ref_trans = np.average(fh5[f'{sn}/attrs/transmitted'])   #### this should be given externally
        trans_data = fh5[f'{sn}/attrs/transmitted'][:Iqphi.shape[0]]/ref_trans    # issue with transmitted data being too long
        qgrid = fh5[f'{sn}/qphi'].attrs['xc']

        if qphi_grp=="merged":
            dm = fh5[f'{sn}/qphi/merged/d'][0]
            Np = int(dm.shape[0]/2)
            w0 = np.zeros_like(dm, dtype=np.int8)
            w0[~np.isnan(dm)] = 1
            w = w0 + np.vstack([w0[Np:,:], w0[:Np,:]])    
            # apply symmetry
            Iqphi1 = da.nansum(da.array([Iqphi, da.concatenate([Iqphi[:,Np:,:], Iqphi[:,:Np,:]], axis=1)]), axis=0) / w        
        else: 
            Iqphi1 = Iqphi
        
        idx = ((qgrid>=q_range[0])&(qgrid<=q_range[1]))
        if dezinger:
            Iphi = da.nanmedian(Iqphi1[:,:,idx], axis=-1)
        else:
            Iphi = da.nanmean(Iqphi1[:,:,idx], axis=-1)

        if qphi_grp=="merged":  # ignoring trans_cor
            idx = (trans_data>0)
            Iphi[idx,:] = (Iphi[idx,:].T/trans_data[idx]).T  

        # the compute needs to be done before the h5 file is closed
        with ProgressBar():
            ret = np.array(Iphi.compute())
                
    return ret

def get_phi_data_dask(dt, q_range, sub_key, 
                      ext="", bkg_thresh=0.6e-2, save_to_overall=True, qphi_grp="merged",
                      dezinger=True, trans_cor=True, ref_trans=5000, debug=False):
    """ 
    """
    Iphis = {sn:bin_phi_data_dask(dt.fn, sn, q_range, dezinger, trans_cor, ref_trans, qphi_grp) for sn in dt.samples}
    nm = f"Iphi{ext}"
    if save_to_overall:
        sns = ['overall']
    else:
        sns = dt.samples
    
    for sn in sns:
        if sn=='overall':
            data = da.concatenate([Iphis[_] for _ in sorted(dt.samples)])
        else:
            data = Iphis[sn]
        print("saving data for ", sn, nm, sub_key)
        #with ProgressBar():
        #    data.to_hdf5(dt.fn, f"{sn}/{nm}/{sub_key}")
        #dt.set_h5_attr(f"{sn}/{nm}", "type", "ndarray")
        dt.add_proc_data(sn, nm, sub_key, data)
        dt.save_data(sn, nm, sub_key)
        dt.set_h5_attr(f"{sn}/{nm}/{sub_key}", "q_range", q_range) 
        dt.set_h5_attr(f"{sn}/{nm}/{sub_key}", "phigrid", dt.phigrid) 

    
def get_phi_data(dt, q_range=[0.01, 2.5], bkg_q_range=None,
                 sub_key="merged", ext="", bkg_thresh=0.6e-2, save_to_overall=True,
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

        dt.add_proc_data(sn, nm, sub_key, data)
        dt.save_data(save_sns=sn, save_data_keys=nm, save_sub_keys=sub_key)
        dt.set_h5_attr(f"{sn}/{nm}/{sub_key}", "phigrid", phi) 

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
            q = dt.get_h5_attr(f"{sn}/Iq/{int_data}", 'qgrid')
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
    
