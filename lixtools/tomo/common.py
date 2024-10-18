import numpy as np
import tomopy
from matplotlib.widgets import Slider,Button
#from lixtools.hdf import h5xs_scan

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
    dsin = dt.proc_data['overall']['maps'][map_key]
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
        dt.set_h5_attr("overall/maps", "rot_cen", c_slider.val)
        plt.draw()
   
    c_slider.on_changed(update)
    c_save.on_clicked(save_cen)
    plt.show()

    return c_slider,c_save

