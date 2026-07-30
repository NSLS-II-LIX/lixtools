import numpy as np
import pylab as plt
import cv2
import h5py
from scipy import ndimage
from lixtools.hdf import h5xs_an,h5xs_scan


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

def plot_data(dt: type(h5xs_scan), data_key, sub_keys=None, auto_scale=False, max_w=10, max_h=4, Nx_min=0, 
              default_cmap='binary', cust_cmaps={}, cmax={}, cmin={}, cm_scale=1.,
              space=0, aspect='auto', sample_in_row=True, roi_w=0, roi_h=0,
              use_alpha_for_all=False, alpha_key="int_cell_Iq", alpha_cutoff=0.03, flat_alpha=False, save_fn=None, dpi=300):
    """
        cust_cmaps a dictionary that specifies sub_keys that should use customized colormaps
        use Nx_min to control the span of the map, if the maps plotted together have different sizes
    """
    if isinstance(dt, list):
        data0 = [t.proc_data['overall'][data_key] for t in dt]
    else:
        data0 = [dt.proc_data[sn][data_key] for sn in dt.samples]
    
    if sub_keys is None:
        sub_keys = dt.proc_data[sn][data_key].keys()
    data = []
    map_size_pixel = {}
    map_size_physical = {}
    
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
            if Nx_min>Nx0 and data_key=="tomo":
                pw = int((Nx_min-Nx0)/2)
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


def make_video(mms, fn, labels=None, fps=3, figsize=(4,4), axis_off=False, **kwargs):
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
        if isinstance(mms[i], MatrixWithCoords):
            mms[i].plot(ax=ax, **kwargs)
        else:
            ax.imshow(mms[i], **kwargs)
        if axis_off:
            plt.axis('off')
        #plt.subplots_adjust(top=0.85, right=0.85)
        ax.set_title(labels[i], y=1.1)
        plt.tight_layout()
        #plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.94, bottom=0.05)
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


def make_video_multimap(sn, datakey, map_list=None, fr_labels=None,
                        color_scale_max={}, sino_scale_y=2., ncols=2, fps=3, **kwargs):
    """ make videos of sinograms or tomograms from multiple contrast channels
        sn: sample name fpr a file generated by stack_2d_slice.py, f"{sn}-3d.h5"
        datakey: maps or tomo
        map_list: contrast channels to be included, could be None, then all are included
        #color_dict: specify if the range of the color needs to be scaled, 
        #    e.g. {"absorption": 0.2} would use 0.2x max value as upper limit of the color scale
        color_scale_max: specify if alternative value to replace max from the data 
        sino_scale_y: step size between slices compared to the step size within a slice 
        ncols: number of columns in the plot, nrows will be calculated based on length of map_list
    """
    dm = {}
    with h5py.File(f"{sn}-3d.h5", "r") as fh:
        keys = set(fh[f'overall/tomo'].keys()) & set(fh[f'overall/maps'].keys())
        if map_list is None:
            map_list = list(keys)
        else:
            map_list = list(keys & set(map_list))
        for k in map_list:
            dm[k] = fh[f'overall/{datakey}/{k}/d'][...]
        phi_list = fh['overall']['maps'].attrs['yc']   

    # color scale
    max_dict = {k:np.nanmax(dm[k]) for k in dm.keys()}
    #for k in color_dict.keys():
    #    max_dict[k] *= color_dict[k]
    for k in color_scale_max.keys():
        max_dict[k] = color_scale_max[k]
    
    # set up the plot 
    w0 = 1.
    shp = dm[map_list[0]].shape
    if datakey=='maps':
        asp = sino_scale_y
        nfr = shp[1]
        h0 = asp*w0*shp[0]/shp[-1]+0.08
        fr_labels = [f'proj={ph:.2f}' for ph in phi_list]
    else:
        nfr = shp[0]
        asp = 1
        h0 = w0+0.1
        if fr_labels is None:
            fr_labels = ["" for _ in range(nfr)]
        if len(fr_labels)!=nfr:
            print(f"mismatched labels against number of frames ({nfr}) ...")
            fr_labels = ["" for _ in range(nfr)]
    nrows = int((len(map_list)-1)/ncols)+1
    
    ww = w0*ncols
    hh = h0*nrows
    sc = 8./np.sqrt(ww*ww+hh*hh)
    w0 *= sc
    h0 *= sc
    
    current_backend = plt.get_backend()
    plt.close('all')
    plt.switch_backend('agg')
    vidwriter = None    
    frames = []

    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
    else: 
        cmap='RdBu_r'

    if 'norm' in kwargs.keys():
        norm = kwargs['norm']
    else: 
        norm="linear"
    
    for fr in range(nfr):
        print(f"processing frame #{fr} ...      \r", end="")
        fig,axs = plt.subplots(figsize=(w0*ncols, h0*nrows), nrows=nrows, ncols=ncols)
        fig.suptitle(fr_labels[fr], y=1)
        if ncols==1:
            axs = np.expand_dims(axs, axis=1)
        elif nrows==1:
            axs = np.expand_dims(axs, axis=0)
                        
        for i in range(nrows):
            for j in range(ncols):
                n = i*ncols+j
                if n<len(map_list):
                    k = sorted(map_list)[n]
                    if datakey=="maps":
                        axs[i][j].imshow(dm[k][:,fr,:], aspect=asp, clim=[0, max_dict[k]], cmap=cmap, norm=norm)
                        axs[i][j].text(1.5, -1.5, k)
                    else:
                        axs[i][j].imshow(dm[k][fr,:,:], aspect=asp, clim=[0, max_dict[k]], cmap=cmap, norm=norm)
                        axs[i][j].text(1.5, -5, k)
                
                axs[i][j].set_axis_off()
        
        plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.94, bottom=0.05)
        # this may be slower
        buf = io.BytesIO()
        fig.savefig(buf, dpi=200)
        buf.seek(0)
        img = PIL.Image.open(buf)
    
        frames.append(img.copy())
        plt.close()
        
    print()
    plt.switch_backend(current_backend)    

    height,width,layers = np.array(frames[0]).shape
    vidwriter = cv2.VideoWriter(f"{sn}-{datakey}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height)) 
    for img in frames:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        vidwriter.write(frame)
    cv2.destroyAllWindows()
    vidwriter.release()        
                

def make_video_from_h5(fn, det_name="camES2", fps=3, figsize=(4,4), 
                       x1=0, x2=-1, y1=0, y2=-1, **kwargs):
    """ fn is the a h5 file packed from experimental data
        assume the data is under f"primary/data/{det_name}_image"
        kwargs are passed onto MatrixWithCorrds.plot()
    """
    current_backend = plt.get_backend()
    plt.close('all')
    plt.switch_backend('agg')
    
    with h5py.File(fn, "r") as fh:
        sn = list(fh.keys())[0]
        images = fh[f'{sn}/primary/data/{det_name}_image'][...]
    nfr = len(images)
    
    vidwriter = None    
    frames = []
    for i in range(nfr):
        fig = plt.figure()
        plt.imshow(images[i])
        plt.axis('off')
        plt.tight_layout()
        
        # this may be slower
        buf = io.BytesIO()
        fig.savefig(buf, dpi=100)
        buf.seek(0)
        img = PIL.Image.open(buf)
        
        frames.append(img.copy())
        plt.close()

    plt.switch_backend(current_backend)

    height,width,layers = np.array(frames[0]).shape
    vidwriter = cv2.VideoWriter(fn.replace(".h5",".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height)) 
    for img in frames:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        vidwriter.write(frame)
    cv2.destroyAllWindows()
    vidwriter.release()    
