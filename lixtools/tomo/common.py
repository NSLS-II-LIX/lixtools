import numpy as np
import tomopy
import pylab as plt
from matplotlib.widgets import Slider,Button

from scipy.ndimage import shift
from scipy.signal import medfilt2d

from skimage.transform import radon
from skimage.filters import threshold_otsu

def img_smooth(img, kernal_size, axis=0):
    ''' from Mingyuan Ge
    '''    
    s = img.shape
    if len(s) == 2:
        img_stack = img.reshape(1, s[0], s[1])
    else:
        img_stack = img.copy()

    if axis == 0:
        for i in range(img_stack.shape[0]):
            img_stack[i] = medfilt2d(img_stack[i], kernal_size)
    elif axis == 1:
        for i in range(img_stack.shape[1]):
            img_stack[:, i] = medfilt2d(img_stack[:,i], kernal_size)
    elif axis == 2:
        for i in range(img_stack.shape[2]):
            img_stack[:, :, i] = medfilt2d(img_stack[:,:, i], kernal_size)
    return img_stack    
    
def recon_mlem(args):
    ''' from Mingyuan Ge
    sino: 
        has a shape of e.g.,(n_angle, FOV_size)
    '''
    #k, sino, theta, rot_cen, num_iter=20, init_recon=None
    k,sino,theta,rot_cen,num_iter = args
    init_recon=None
    if not init_recon is None:
        if len(init_recon.shape) == 2:
            tmp = np.expand_dims(init_recon, axis=0)
        else:
            tmp = init_recon
        rec = tomopy.recon(np.expand_dims(sino, axis=1), theta, rot_cen, algorithm='mlem', num_iter=num_iter, init_recon=tmp)
    rec = tomopy.recon(np.expand_dims(sino, axis=1), theta, rot_cen, algorithm='mlem', num_iter=num_iter)
    return k,rec

def otsu_mask(img, kernal_size, iters=1, bins=256, erosion_iter=0):
    ''' from Mingyuan Ge
    '''
    img_s = img.copy()
    img_s[np.isnan(img_s)] = 0
    img_s[np.isinf(img_s)] = 0
    for i in range(iters):
        img_s = img_smooth(img_s, kernal_size)
    thresh = threshold_otsu(img_s, nbins=bins)
    mask = np.zeros(img_s.shape)
    #mask = np.float32(img_s > thresh)
    mask[img_s > thresh] = 1
    mask = np.squeeze(mask)
    if erosion_iter:
        struct = ndimage.generate_binary_structure(2, 1)
        struct1 = ndimage.iterate_structure(struct, 2).astype(int)
        mask = ndimage.binary_erosion(mask, structure=struct1).astype(mask.dtype)
    mask[:erosion_iter+1] = 1
    mask[-erosion_iter-1:] = 1
    mask[:, :erosion_iter+1] = 1
    mask[:, -erosion_iter-1:] = 1
    return mask  

def calc_tomo(args):
    an,mm,kwargs = args
    
    # filter out incomplete data
    idx = ~np.isnan(mm.d.sum(axis=1))
    dmap = mm.d[idx, :]
    yc = mm.yc[idx]
    xc = mm.xc
    proj = dmap.reshape((len(yc),1,len(xc)))
    
    if "algorithm" in kwargs.keys():
        algorithm = kwargs.pop("algorithm")
    else:
        algorithm = "gridrec"
        
    if "center" in kwargs.keys():
        rot_center = kwargs.pop("center")
    else:
        rot_center = tomopy.find_center(proj, np.radians(yc))
    
    recon = tomopy.recon(proj, np.radians(yc), center=rot_center, algorithm=algorithm, sinogram_order=False, **kwargs)
    recon = tomopy.circ_mask(recon, axis=0) #, ratio=0.95)
    
    return [an,recon[0,:,:]]


def cen_test(dt, map_key="absorption", test_range=30, clim=[], cmap='bone'):    
    """ adapted from Mingyuan Ge
        dt: type(h5xs_scan), omitted for now to avoid circular reference
    """
    if len(dt.h5xs)>1:
        sn = "overall"
    else:
        sn = dt.samples[0]

    dsin = dt.proc_data[sn]['maps'][map_key]
    sino = dsin.d
    theta = np.radians(dsin.yc)
    prj = np.expand_dims(sino, axis=1)
    s = prj.shape # e.g., [180, 1, 640]
    start = int(s[2] / 2 - test_range)
    stop = int(s[2] / 2 + test_range)
    steps = stop-start + 1
    
    cen = np.linspace(start, stop, steps)
    img = np.zeros([len(cen), s[2], s[2]])
    
    for i in range(len(cen)):
        img[i] = tomopy.recon(prj, theta, center=cen[i], algorithm="gridrec")
    img = tomopy.circ_mask(img, axis=0, ratio=1.0) #0.8)

    fig, ax = plt.subplots()
    axis = 0
    index_init = int(img.shape[axis]//2)
        
    if len(clim) == 2:
        im = ax.imshow(img.take(index_init,axis=axis), cmap=cmap, clim=clim, origin="lower")
    else:
        im = ax.imshow(img.take(index_init,axis=axis), cmap=cmap, origin="lower")  
 
    fig.subplots_adjust(bottom=0.15)
    axslide = fig.add_axes([0.1, 0.03, 0.65, 0.03])
    c_slider = Slider(ax=axslide, label='center', valmin=cen[0], valmax=cen[-1], 
                       valstep=1, valinit=cen[index_init])
    axsave = fig.add_axes([0.85, 0.03, 0.1, 0.03])
    c_save = Button(axsave, "save")
    
    def update(val):
        im.set_data(img.take(val-cen[0],axis=axis))
        fig.canvas.draw_idle()
        
    def save_cen(event):
        print(f"saving rot_cen: {c_slider.val}")
        dt.set_h5_attr(f"{sn}/maps", "rot_cen", c_slider.val)
        plt.draw()
   
    c_slider.on_changed(update)
    c_save.on_clicked(save_cen)
    plt.show()

    return c_slider,c_save


def get_sinomask(dt, ref_key="absorption", abs_cutoff=0.02):
    """ attempt to get a clean mask for sinograms based on a reconstructed tomogram 
    """
    dd = dt.proc_data['overall']['maps'][ref_key].d.copy()
    dd = otsu_mask(dd, 3)
    h = dd.shape[0]
    mask = cv2.blur(np.vstack([dd, dd, dd]), [5,1])
    mask = mask[h:2*h, :]
    mask[mask>abs_cutoff] = 1
    mask[mask<abs_cutoff] = 0

    return mask


def merge_XRF_channels(dt, prefix="xrf_", ref_key="absorption", ref_cutoff=0.02,
                       map_key="maps", new_map_key="maps_ext"):

    map_list = [kn for kn in dt.proc_data['overall'][map_key].keys() if prefix in kn]
    rot_cen = dt.get_h5_attr(f"overall/{map_key}", "rot_cen")
    xc = dt.get_h5_attr(f"overall/{map_key}", "xc")
    offset = len(xc)/2-rot_cen 

    for kn in map_list:
        xrf_data = dt.proc_data['overall'][map_key][kn]
        N = len(xrf_data)
        
        # assume the first half of the channels are the reference
        m1 = xrf_data[0].copy()
        m1.d = shift(np.sum([mm.d for mm in xrf_data[:int(N/2)]], axis=0), (0, offset))
        m2 = xrf_data[0].copy()
        m2.d = shift(np.sum([mm.d for mm in xrf_data[int(N/2):]], axis=0), (0, offset))
        m2.yc = m1.yc-180
        
        mask = get_sinomask(dt)
        bkg1 = np.average(m1.d[mask==0])
        bkg2 = np.average(m2.d[mask==0])
        m1.d -= bkg1
        m1.d[m1.d<0] = 0
        m2.d -= bkg2
        m2.d[m2.d<0] = 0        
        m2.d *= np.sum(m1.d)/np.sum(m2.d)
        m2.d = np.fliplr(m2.d)
        ma = m1.merge([m2])
        
        dt.add_proc_data("overall", new_map_key, kn, ma)
        
    # create an extended, clean version of absorption sinogram for fluorescence intensity correction
    kn = "absorption"
    mabs = dt.proc_data['overall']['tomo'][kn].copy()
    mask = cv2.blur(otsu_mask(mabs.d, 3), [5,5])
    mask[mask>0.02] = 1
    img = mask*mabs.d
    sino = dt.proc_data['overall'][map_key][kn].copy()
    sino.d = radon(img, theta=sino.yc).T

    sino1 = sino.copy()
    sino1.yc = sino.yc-180
    sino1.d = np.fliplr(sino.d)
    ss = sino.merge([sino1])
    
    dt.add_proc_data("overall", new_map_key, kn, ss)
    
    dt.save_data(save_sns=["overall"], save_data_keys=[new_map_key])
    dt.set_h5_attr(f"overall/{new_map_key}", "rot_cen", len(xc)/2)