from lixtools.hdf import h5xs_scan
from lixtools.hdf.scan import calc_tomo
from lixtools.mapping.common import make_map_from_overall_attr
import numpy as np
import multiprocessing as mp
import cv2 as cv
import pylab as plt
from skimage import feature,morphology,filters

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

def calc_CI2(dt: type(h5xs_scan), sn='overall', data_key="tomo", 
            cell_key='int_cell_Iq', amor_key='int_amor_Iq',
            sc=1., amor_cutoff=0.02, show_mask=False)-> None:
    """ sc is needed to account for the intensity difference between cellulose and amorphous 
        components due to the way they are calculated: span of q-range, shaping factor
    """
    mm = dt.proc_data[sn][data_key][cell_key].copy()
    dc = dt.proc_data[sn][data_key][cell_key].d
    da = dt.proc_data[sn][data_key][amor_key].d

    amo1 = np.array(da/np.max(da)*256, dtype=np.uint8)
    amo2 = cv2.adaptiveThreshold(amo1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
    amo2[da<amor_cutoff] = 0
    idx = (amo2>0)
    
    if show_mask:
        plt.figure(figsize=(10,3))
        plt.subplot(131)
        plt.hist(amo.flatten(), bins=100, alpha=0.3)
        plt.yscale('symlog')
        plt.subplot(132)
        plt.imshow(amo1, cmap="binary")
        plt.subplot(133)
        plt.imshow(amo2, cmap="binary")

    da *= sc
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

def prep_mfa_data(dt: type(h5xs_scan), rot_center, data_sub_key, 
                  algorithm="pml_hybrid", num_iter=60, ref_key="absorption", ref_cutoff=0.02, base=0)-> None:
    """ base is subtracted as scattering background 
    """
    #phi = dt.get_h5_attr(f"overall/Iphi/{data_sub_key}", "phigrid")
    phi = dt.phigrid
    idx = ((phi>=-90)&(phi<=90))
    phi0 = phi[idx]
    mms = []
    for i in np.arange(len(phi))[idx]:
        mm = make_map_from_overall_attr(dt, dt.proc_data['overall']['Iphi'][data_sub_key][:, i], 
                                        template_grp=ref_key, map_name=None)
        mms.append(mm)

    dt.proc_data['overall']['maps'][f'Iphi-{data_sub_key}'] = mms
    dt.save_data(save_sns='overall', save_data_keys=['maps'], save_sub_keys=[f'Iphi-{data_sub_key}'])
        
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
    
    dt.proc_data['overall']['tomo'][f'Iphi-{data_sub_key}'] = tms
    dt.proc_data['overall']['tomo'][f'mfa-{data_sub_key}'] = mm    
    dt.save_data(save_sns='overall', save_data_keys=['tomo'], 
                 save_sub_keys=[f'Iphi-{data_sub_key}', f'mfa-{data_sub_key}'])
    dt.set_h5_attr(f"overall/maps/Iphi-{data_sub_key}", "phi", phi0)
    dt.set_h5_attr(f"overall/tomo/Iphi-{data_sub_key}", "phi", phi0)


def segment_EBWP(tms, n_blur=13, n_blur_pith=25, th_bark=130,
                 pith_pix=None, bkg_pix=[0,0],
                 pith_fill_pix_list=[], bark_remove_pix_list=[]):
    """ create masks for empty/bark/wood/pith
        the input should be a collection of maps, e.g. tms = dt.proc_data['overall']['tomo']
        that correspond to the cross-section of a plant stem
    """
    d1 = cv.GaussianBlur(tms['int_amor_Iq'].d, (9,9),0)
    im1 = np.array(d1/np.max(d1)*255, dtype=np.uint8)
    d2 = cv.GaussianBlur(tms['int_cell_Iq'].d, (9,9),0)
    im2 = np.array(d2/np.max(d2)*255, dtype=np.uint8)
    
    cc = np.array(feature.canny(im2)*255, dtype=np.uint8)
    cc = cv.GaussianBlur(cc, (n_blur,n_blur), 0)
    cc[cc>16] = 255
    cc[cc<=16] = 0
    _,cc1,_,_ = cv.floodFill(cc, None, bkg_pix, 127)
    if pith_pix is None:
        pitch_pix = [int(cc1.shape[0]/2)+10,int(cc1.shape[1]/2)]
    _,cc2,_,_ = cv.floodFill(cc1, None, pitch_pix, 64)  # needs adjustment
    
    bkg_mask = np.array((cc2==127), dtype=np.uint8)
    kernel = np.ones((n_blur, n_blur), np.uint8) 
    bkg_mask = cv.morphologyEx(bkg_mask, cv.MORPH_CLOSE, kernel)
    
    cc = np.array(feature.canny(im2)*255, dtype=np.uint8)
    cc = cv.GaussianBlur(cc, (n_blur,n_blur), 0)
    cc[cc>16] = 255
    cc[cc<=16] = 0
    _,cc1,_,_ = cv.floodFill(cc, None, bkg_pix, 127)

    if pith_pix is None:
        pith_pix = [int(cc1.shape[0]/2)+10,int(cc1.shape[1]/2)]
    _,cc2,_,_ = cv.floodFill(cc1, None, pith_pix, 64)  # needs adjustment

    pith_mask = np.array((cc2==64), dtype=np.uint8)
    kernel = np.ones((n_blur_pith, n_blur_pith), np.uint8) 
    pith_mask = cv.morphologyEx(pith_mask, cv.MORPH_CLOSE, kernel)

    for pix in pith_fill_pix_list:
        _,bark_mask,_,_ = cv.floodFill(pith_mask, None, pix, 1)  # pixel position
    
    plt.figure(figsize=(10,3))
    plt.subplot(121)
    _ = plt.hist(im1[(bkg_mask==0)&(pith_mask==0)].flatten(), bins=50)
    plt.plot([th_bark, th_bark], [0, 5000], "k--", label="bark cutoff")
    plt.legend(frameon=False)
    plt.subplot(122)
    plt.imshow(im1, aspect=1)
    
    kernel = np.ones((n_blur, n_blur), np.uint8) 
    bark_mask = np.array((im2>th_bark)&(bkg_mask==0)&(pith_mask==0), dtype=np.uint8)
    bark_mask = cv.morphologyEx(bark_mask, cv.MORPH_CLOSE, kernel)
    bark_mask = cv.morphologyEx(bark_mask, cv.MORPH_OPEN, kernel)
    #bark_mask = morphology.remove_small_objects(bark_mask, min_size=20)

    for pix in bark_remove_pix_list:
        _,bark_mask,_,_ = cv.floodFill(bark_mask, None, pix, 0)  # pixel position
    
    kernel = np.ones((n_blur,n_blur),np.uint8)
    woody_mask = np.array((bkg_mask==0)&(bark_mask==0)&(pith_mask==0), dtype=np.uint8)
    woody_mask = cv.erode(woody_mask, kernel, iterations = 3)
    woody_mask = cv.dilate(woody_mask, kernel, iterations = 2)
    #woody_mask = morphology.remove_small_objects(woody_mask, min_size=20)
    
    plt.figure(figsize=(10,3))
    plt.subplot(141)
    plt.imshow(bkg_mask)
    plt.axis('off')
    plt.subplot(142)
    plt.imshow(pith_mask)
    plt.axis('off')
    plt.subplot(143)
    plt.imshow(woody_mask)
    plt.axis('off')
    plt.subplot(144)
    plt.imshow(bark_mask)
    plt.axis('off')
    
    return {"bkg": bkg_mask, "pith": pith_mask, "wood": woody_mask, "bark": bark_mask}