# Revised from code from Mingyuan. rotation center already found
# method described in "Three-dimensional imaging of grain boundaries via quantitative fluorescence X-ray tomography analysis"
# DOI:10.1038/s43246-022-00259-x
import xraylib
import tomopy,h5py,json
import numpy as np
import multiprocessing as mp

from scipy import ndimage
from scipy.ndimage import shift
from skimage.transform import resize
from scipy.signal import savgol_filter
from matplotlib.widgets import Slider
from skimage import io

from .common import otsu_mask,recon_mlem

def generate_detector_mask(alfa0, theta0, leng0):

    """
    Generate a pyramid shape mask inside a rectangular box matrix.
    Simulating the light transmission from a point source and then collected by rectangular detector

    Parameters:
    -----------

    alfa0:  int
            horizontal dispersion angle, in unit of degree

    theta0: int
            vertial dispersion angle, in unit of degree

    leng0:  int
            radial length of light transmission, in unit of pixels

    Returns:
    --------

    3D array:
        mask profile; matrix elements are zero outside the detection region.
    """

    alfa = np.float32(alfa0) / 2 / 180 * pi
    theta = np.float32(theta0) / 2 / 180 * pi

    N1_0 = np.int16(np.ceil(leng0 * np.tan(alfa))) # for original matrix
    N2_0 = np.int16(np.ceil(leng0 * np.tan(theta)))

    leng = leng0 + 30
    N1 = np.int16(np.ceil(leng * np.tan(alfa)))
    N2 = np.int16(np.ceil(leng * np.tan(theta)))

    Mask = np.zeros([2*N1-1, 2*N2-1, leng])

    s = Mask.shape
    s0= (2*N1_0-1, 2*N2_0-1, leng0) # size of "original" matrix

    M1 = np.zeros((s[0], s[2]))
    M2 = np.zeros((s[1], s[2]))

    M1 = g_mask((s[0], s[2]), alfa, N1)
    M2 = g_mask((s[1], s[2]), theta, N2)
    M1[N1-1,:] = 1
    M1[N1-1,0] = 0
    M2[N2-1,:] = 1
    M2[N2-1,0] = 0

    Mask1 = Mask.copy()
    Mask2 = Mask.copy()

    for i in range(s[1]):
        Mask1[:,i,:] = M1
    for i in range(s[0]):
        Mask2[i,:,:] = M2

    Mask = Mask1 * Mask2 # element by element multiply
    shape_mask = Mask > 0

    a,b,c = np.mgrid[1:s[0]+1, 1:s[1]+1, 1:s[2]+1]
    dis = np.sqrt((a-N1)**2 + (b-N2)**2 + (c-1)**2)

    dis[N1-1,N2-1,0]=1
    dis = dis * shape_mask * 1.0
    a = np.floor(dis)  # integer part of "dis"
    b = dis - a   # decimal part of "dis"
    #M_normal = np.zeros(Mask.shape)*shape_mask

    M_normal = g_radial_mask(Mask.shape, dis, shape_mask) # generate mask with radial distance

    Mask3D = M_normal * Mask

    cent = np.array([np.floor(s[0]/2), np.floor(s[1]/2)])
    delt = np.array([np.floor(s0[0]/2), np.floor(s0[1]/2)])

    xs = np.int16(cent[0]-delt[0])
    xe = np.int16(cent[0]+delt[0]+1)
    ys = np.int16(cent[1]-delt[1])
    ye = np.int16(cent[1]+delt[1]+1)
    zs = 0
    ze = leng0

    Mask3D_cut = Mask3D[xs:xe,ys:ye,zs:ze] # col, slice, row
    Mask3D_cut = np.transpose(Mask3D_cut, [1,2,0]) # slice, row, col
    Mask3D_cut[np.isnan(Mask3D_cut)] = 0

    return Mask3D_cut


def prep_detector_mask3D(xrf_detector_distance=50, detector_size=8.4,
                         length_maximum=200, fn_save='mask3D.h5'):
    print('Generating detector 3D mask ...')

    
    alfa = detector_size/xrf_detector_distance * 180 /np.pi
    theta = detector_size/xrf_detector_distance * 180 /np.pi

    mask = {}
    for i in trange(length_maximum, 6, -1):
        mask[f'{i}'] = generate_detector_mask(alfa, theta, i)

    with h5py.File(fn_save, 'w') as hf:
        for i in range(7, length_maximum+1):
            k = f'{i}'
            hf.create_dataset(k, data=mask[k])
    return mask    
    
def load_mask3D(fn='mask3D.h5'):
    f = h5py.File(fn, 'r')
    keys = f.keys()
    mask3D = {}
    for k in keys:
        mask3D[k] = np.array(f[k])
    return mask3D


#def generate_H(elem_type, ref3D_tomo, sli, angle_list, attn_data, bad_angle_index=[], flag=1):
def generate_H(ref3D_tomo, sli, angle_list, attn_data, bad_angle_index=[], flag=1):
    """
    Generate matriz H and I for solving eqution: H*C=I
    In folder of file_path:
            Needs 3D attenuation matrix at each rotation angle
    e.g. H = generate_H('Gd', Gd_tomo, 30, angle_list, bad_angle_index, file_path='./Angle_prj', flag=1)

    Parameters:
    -----------

    elem_type: chars
        e.g. elem_type='Gd'
    ref3D_tomo: 3d array
        a referenced 3D tomography data with same shape of attentuation matrix
    sli: int
        index of slice ID in 3D tomo
    angle_list: 1d array
        rotation angles in unit of degree
    attn_data[element][prj_angle]: 
        attenuation matrix with name of:  e.g. atten_Gd_prj_50.0.h5
        these files can be generated through function of: 'write_attenuation'
    bad_angle_index: 1d array
        angle_index that angle will not be used,
        e.g. bad_angle_index=[0,10,36] --> angle_list[0] will not be used
    flag: int
        flag = 1: use attenuation matrix read from file
        flag = 0: generate non-attenuated matrix (Radon matrix)

    Returns:
    --------
    2D array
    """

    ref_tomo = ref3D_tomo.copy()
    theta = np.array(angle_list / 180 * np.pi)
    num = len(theta) - len(bad_angle_index)
    s = ref_tomo.shape
    cx = (s[2]-1) / 2.0       # center of col
    cy = (s[1]-1) / 2.0       # center of row
 
    H_tot = np.zeros([s[2]*num, s[2]*s[2]])
    k = -1
    for i in range(len(theta)):
        print(f"processing angle {i}    \r", end="")
        if i in bad_angle_index:
            continue
        k = k + 1
        if flag:
            #att = attn_data[elem_type][sli]
            att = attn_data[sli]
            if len(att.shape) == 3:
                att = att[sli]
        else:
            att = np.ones([s[1],s[2]])

        T = np.array([[np.cos(-theta[i]), -np.sin(-theta[i])],[np.sin(-theta[i]), np.cos(-theta[i])]])
        H = np.zeros([s[2], s[1]*s[2]])
        for col in range(s[2]):
            for row in range(s[1]):
                p = row
                q = col
                cord = np.dot(T,[[p-cx],[q-cy]]) + [[cx],[cy]]
                if ((cord[0] > s[1]-1) or (cord[0] <= 0) or (cord[1] > s[2]-1) or (cord[1] <= 0)):    continue
                r_frac = cord[0] - np.floor(cord[0])
                c_frac = cord[1] - np.floor(cord[1])
                r_up = int(np.floor(cord[0]))
                r_down = int(np.ceil(cord[0]))
                c_left = int(np.floor(cord[1]))
                c_right = int(np.ceil(cord[1]))

                ul = r_up * s[2] + c_left
                ur = r_up * s[2] + c_right
                dl = r_down * s[2] + c_left
                dr = r_down * s[2] + c_right

                if (r_up >= 0 and c_left >=0):
                    H[q, ul] = H[q, ul] + att[p, q] * (1-r_frac) * (1-c_frac)
                if (c_left >=0):
                    H[q, dl] = H[q, dl] + att[p, q] * r_frac * (1-c_frac)
                if (r_up >= 0):
                    H[q, ur] = H[q, ur] + att[p,q] * (1-r_frac) * c_frac
                H[q, dr] =  H[q, dr] + att[p, q] * r_frac * c_frac
        H_tot[k*s[2] : (k+1)*s[2], :] = H

    return H_tot


#def generate_I(elem_type, ref3D_tomo, sli, angle_list, prj_data, bad_angle_index=[]):
def generate_I(ref3D_tomo, sli, angle_list, prj_data, bad_angle_index=[]):
    """
    Generate matriz I for solving eqution: H*C=I
    In folder of file_path:
            Needs aligned 2D projection at each rotation angle
    e.g. I = generate_I(Gd, 30, angle_list, bad_angle_index, file_path='./Angle_prj')

    Parameters:
    -----------

    elem_type: chars
        e.g. elem_type='Gd'
    ref3D_tomo: 3d array
        a referenced 3D tomography data with same shape of attentuation matrix
    sli: int
        index of slice ID in 3D tomo
    angle_list: 1d array
        rotation angles in unit of degree
    bad_angle_index: 1d array
        angle_index that angle will not be used,
        e.g. bad_angle_index=[0,10,36] --> angle_list[0] will not be used
    file_path: folder path. Under the path, it includes:
         aligned projection image:         e.g. Gd_ref_prj_50.0.h5
         these files can be generated through function of: 'write_projection'

    Returns:
    --------
    1D array

    """
    theta = np.array(angle_list / 180 * np.pi)
    num = len(theta) - len(bad_angle_index)
    s = ref3D_tomo.shape
    I_tot = np.zeros(s[2]*num)
    k = -1
    for i in range(len(theta)):
        print(f"processing angle {i}    \r", end="")
        if i in bad_angle_index:
            continue
        k = k + 1
        prj = prj_data[sli][i]
        #prj = prj_data[f"xrf_{elem_type}"][sli][i]
        I_tot[k*s[2] : (k+1)*s[2]] = prj

    return I_tot

def mlem_matrix(img2D, p, y, iter_num=10):
    img2D = np.array(img2D)
    img2D[np.isnan(img2D)] = 0
    img2D[img2D < 0] = 0

    A_new = img2D.flatten()    # convert 2D array in 1d array
    for n in range(iter_num):
        print(f'iteration: {n}           \r', end="")
        Pf = p @ A_new
        Pf[Pf < 1e-6] = 1
        t1 = p
        t2 = y.flatten() / Pf.flatten()
        t2 = np.reshape(t2, (len(t2), 1))
        a_sum = np.sum(t1*t2, axis=0)
        b_sum = np.sum(p, axis=0)
        a_sum[b_sum<=0] = 0
        b_sum[b_sum<=0] = 1
        A_new = A_new * a_sum / b_sum
        A_new[np.isnan(A_new)] = 0
    img_cor = np.reshape(A_new, img2D.shape)
    return img_cor

def plot_image_stack(data, cvalues, axis=0, index_init=None, clim=[]):
    fig, ax = plt.subplots()
    if index_init is None:
        index_init = int(data.shape[axis]//2)
        
    if len(clim) == 2:
        im = ax.imshow(data.take(index_init,axis=axis), cmap='bone', clim=clim)
    else:
        im = ax.imshow(data.take(index_init,axis=axis), cmap='bone')
    fig.subplots_adjust(bottom=0.15)
    axslide = fig.add_axes([0.1, 0.03, 0.8, 0.03])
    im_slider = Slider(
        ax=axslide,
        label='index',
        valmin=cvalues[0],
        valmax=cvalues[-1],
        valstep=1,
        valinit=cvalues[index_init],
    )
    def update(val):
        im.set_data(data.take(val-cvalues[0],axis=axis))
        fig.canvas.draw_idle()
   
    im_slider.on_changed(update)
    plt.show()
    return im_slider  
    
def rm_nan(*args):

    """
    Remove nan and inf in data
    e.g. a =  rm_nan(data1, data2, data3)

    Parameters:
    -----------
    args: a list of ndarray data with same shape
    """

    num = len(args)
    s = args[0].shape

    data = np.zeros([num] + list(s))
    for i in range(num):
        data[i] = args[i]
    data = np.array(data)
    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    data = data[0]

    return np.array(data)

def rot3D(img_raw, rot_angle):

    """
    Rotate 2D or 3D or 4D(set of 3D) image with angle = rot_angle
    rotate anticlockwise

    Parameters:
    -----------
    img:        2D or 3D array or 4D array

    rot_angle:  float
                rotation angles, in unit of degree

    Returns:
    --------
    2D or 3D or 4D array with same shape of input image
        all pixel value is large > 0

    """

    img = np.array(img_raw)
    img = rm_nan(img)
    s = img.shape
    if len(s) == 2:    # 2D image
        img_rot = ndimage.rotate(img, rot_angle, reshape=False)
    elif len(s) == 3:  # 3D image, rotating along axes=0
        img_rot = ndimage.rotate(img, rot_angle, axes=[1,2], reshape=False)
    elif len(s) == 4:  # a set of 3D image
        img_rot = np.zeros(img.shape)
        for i in range(s[0]):
            img_rot[i] = ndimage.rotate(img[i], rot_angle, axes=[1,2], reshape=False)
    else:
        raise ValueError('Error! Input image has dimension > 4')
    img_rot[img_rot < 0] = 0

    return img_rot


def retrieve_data_mask(data3d, row, col, sli, mask):
    """
    Retrieve data defined by mask, orignated at position(sli/2, row, col/2),
    and then multiply it by mask, and then take the sum

    Parameters:
    -----------
    data: 3d array

    row, col, sli: int
          (sli/2, row, col/2) is the original of the position to retrieve data

    mask: 3d array
          shape of mask should be smaller than shape of data

    Returns:
    --------
    3D array:  data defined by mask-shape multiplied by mask

    """
    s0 =np.int16(data3d.shape)
    s = np.int16(mask.shape)
    xs = int(row)
    xe = int(row + s[1])
    ys = int(col - np.floor(s[2]/2))
    ye = int(col + np.floor(s[2]/2)+1)
    zs = int(sli - np.floor(s[0]/2))
    ze = int(sli + np.floor(s[0]/2)+1)
    ms = mask.shape
    m_xs = 0;   m_xe = ms[1];
    m_ys = 0;   m_ye = ms[2];
    m_zs = 0;   m_ze = ms[0];
    if xs < 0:
        m_xs = -xs
        xs = 0
    if xe > s0[1]:
        m_xe = s0[1] - xe + ms[1]
        xe = s0[1]
    if ys < 0:
        m_ys = -ys
        ys = 0
    if ye > s0[2]:
        m_ye = s0[2] - ye + ms[2]
        ye = s0[2]
    if zs < 0:
        m_zs = -zs
        zs = 0
    if ze > s0[0]:
        m_ze = s0[0] - ze + ms[0]
        ze = s0[0]
    data = np.sum(data3d[zs:ze, xs:xe, ys:ye] * mask[m_zs:m_ze, m_xs:m_xe, m_ys:m_ye])
    return data


def atten_2D_slice(sli, mu_ele, mask3D, display_flag=True):
    # calculate pixel attenuation length
    if display_flag:
        print(f'sli={sli}   \r', end="")
    s_mu = mu_ele.shape
    atten_ele = np.ones([s_mu[1], s_mu[2]])
    #sli = int(s_mu[0]/2)
    for i in range(s_mu[1]): # row
        length = max(s_mu[1] - i, 7)
        mask = np.asarray(mask3D[f'{length}'], dtype='f4')
        for j in np.arange(0, s_mu[2]): # column
            #if mu_ele[sli, i, j] == 0:
            #    continue
            atten_ele[i, j] = retrieve_data_mask(mu_ele, int(i), int(j), sli, mask)
    return atten_ele


def calc_attn(args):
    ele,mu,mub,angle_list,s,mask3D,max_thick,pix = args
    n_angle = len(angle_list)
    ret = []
    for i in range(n_angle):
        '''
        consider the relationship between ndimage_rotation and tomopy implementation, we need to rotate negative angle
        In experiment configuration, incident x-ray is from top-down, xrf detector is at right side
        In FL-correction implementation, xrf detector is assumed at the bottom, so we need to rotate -90 degrees
        In calculating incident x-ray attenuation, FL-correction assume the x-ray is from top-down, so we don't need additional rotation
        '''
        print(f"processing angle {i} for {ele} ...    \r", end="")
        # step 1: calculate xrf attenuation
        mu_xrf_angle = rot3D(mu, -angle_list[i])

        mu_xrf_angle_rot = rot3D(mu_xrf_angle, -90)            
        atten_xrf = atten_2D_slice(max_thick//2, mu_xrf_angle_rot, mask3D, display_flag=False)
        atten_xrf = np.exp(-atten_xrf * pix)
        atten_xrf = rot3D(atten_xrf, 90)  # rotate back to exprimental configuration

        # step 2: calculate incident x-ray attenuation
        atten_x_ray = np.ones((s[1], s[2]))   # (340, 340)
        mu_x_angle = rot3D(mub[0], -angle_list[i]) 

        for j in range(1, s[1]):  # number of rows, from top to down 
            atten_x_ray[j] = atten_x_ray[j-1] * np.exp(-mu_x_angle[j] * pix) 

        atten_comb = atten_x_ray * atten_xrf # (85, 85)
        atten_comb = np.expand_dims(atten_comb, axis=0)
        ret.append(atten_comb)
    
    return ele,ret

def calc_img_cor(args):
    ele,rec_bin,angle_list,attn,sino_bin,mask_bin,iter_num = args
    print(f"running H for {ele} ...     \r", end="")
#    H = generate_H(ele, rec_bin, 0, angle_list, attn, [])
    H = generate_H(rec_bin, 0, angle_list, attn[ele], [])
    print(f"running I for {ele} ...     \r", end="")
#def generate_I(elem_type, ref3D_tomo, sli, angle_list, prj_data, bad_angle_index=[]):
#prj_data[f"xrf_{elem_type}"]
#    I = generate_I(ele, rec_bin, 0, angle_list, sino_bin, [])
    I = generate_I(rec_bin, 0, angle_list, sino_bin[f"xrf_{ele}"], [])
    img_cor = mlem_matrix(np.ones(rec_bin.shape)*mask_bin, H, I, iter_num=iter_num)  
    
    return ele,img_cor


def run_abs_cor(dt, mask3D, pixel_size=0.005, binning=4,
                map_for_mask='int_cell_Iq', incident_energy=15.138, matrix_chem_form='C6H10O5'):
    """ pixel_size in mm
        matrix_chem_form: chemical formula for the matrix material, for absorption calculation
    """
    pix_bin1 = pixel_size*0.1   # use cm for some reason
    pix = binning * pix_bin1
    
    ele_list = json.loads(dt.get_h5_attr("/overall/XRF/basis", "ele_list"))
    em_E = {}
    ebins = np.arange(3.1, 11.0, 0.01)
    for i,line in enumerate(ele_list):
        em_E[line.split("_")[0]] = np.average(ebins, weights=dt.proc_data['overall']['XRF']['basis'][i])
    element_list = em_E.keys()
    
    mks1 = [f'xrf_{ele}' for ele in element_list]
    mks2 = [f'xrf_{ele}_cor' for ele in element_list]
    
    sino_norm = dt.proc_data['overall']['maps']['absorption'].d
    l = sino_norm.shape[-1]
    sino_norm = sino_norm[:, :l//2*2] # make image size an even number
    sino_norm[sino_norm<0] = 0
    sino_norm_f = mf(sino_norm, 5)
    diff = sino_norm - sino_norm_f
    idx = np.abs(diff) > 0.02

    sino_norm[idx] = sino_norm_f[idx]
    ang_list = dt.proc_data['overall']['maps']['absorption'].yc
    theta_list = np.radians(ang_list)
    
    rot_cen = dt.get_h5_attr("overall/maps", "rot_cen")

    '''
        shift the sinogram if the rotation center is not in the middle of the image
        reconstruction on all elements and transmission signal
    '''
    # calculate the amount of pixel-shift to move the rotation center to the center of image
    shift_pixels = sino_norm.shape[1] / 2 - 0.5 - rot_cen
    rc = rot_cen + shift_pixels # it is equavalent to sino_norm.shape[1] / 2 - 0.5

    sino = {}
    sino_bin = {}
    rec = {}
    rec_bin = {}

    sino['trans'] = dt.proc_data['overall']['maps']['absorption'].d
    for ele in element_list:
        if isinstance(dt.proc_data['overall']['maps'][f'xrf_{ele}'], list):
            sino[f"xrf_{ele}"] = dt.proc_data['overall']['maps'][f'xrf_{ele}'][0].d + dt.proc_data['overall']['maps'][f'xrf_{ele}'][1].d
        else:
            sino[f"xrf_{ele}"] = dt.proc_data['overall']['maps'][f'xrf_{ele}'].d

    print("calculate initial tomograms ...")
    pool = mp.Pool(len(element_list)+1)   # + absorption
    jobs = []
    for k in sino.keys():
        sino[k] = np.expand_dims(shift(sino[k], (0, shift_pixels), order=3), axis=0)
        jobs.append(pool.map_async(recon_mlem, [(k, sino[k][0], theta_list, rc, 20)]))
        print(f"started job for {k}             \r", end="")

    pool.close()
    for job in jobs:
        k,data = job.get()[0]
        rec[k] = data
        ss = sino[k].shape # (1, 121, 480), (1, 121, 680)
        st = rec[k].shape # (1, 480, 480), (1, 680, 680)
        sino_bin[k] = resize(sino[k], (1, ss[1], ss[2]//binning), anti_aliasing=True)
        rec_bin[k] = resize(rec[k], (1, st[1]//binning, st[2]//binning), anti_aliasing=True)   
        print(f"got data for {k}             \r", end="")
    pool.join()
    print("done ...                 ")

    # generate mask
    #m = otsu_mask(rec['trans'], 11)
    m = otsu_mask(dt.proc_data['overall']['tomo'][map_for_mask].d, 1)
    m = np.expand_dims(m, axis=0)
    m_bin = resize(m, (1, st[1]//binning, st[2]//binning), anti_aliasing=False)
    m_bin[m_bin>0.2] = 1    
    
    # Energy of incident x-ray 
    x_E = incident_energy

    # we name the polymer as "base"
    base = matrix_chem_form # 'C16H14O3'   # cellulose is supposed to be (C6H10O5)n
    rho_base = 1.0 # g/cm3

    cs = {}
    for key in em_E.keys():
        cs[f'base-{key}'] = xraylib.CS_Total_CP(base, em_E[key])
    cs['base-x'] = xraylib.CS_Total_CP(base, x_E) # absorption cross-section for incident x-ray

    s_rec = rec['trans'].shape 

    # calculate the density of the polymer matrix according to measured "transimission" tomography
    rho_2D = rec['trans'] / (cs['base-x'] * pix_bin1) # (1, 680, 680)
    rho_2D_bin = resize(rho_2D, (1, s_rec[-2]//binning, s_rec[-1]//binning))

    # read rotation angles. Note that: reconstruction in tomopy, it is clockwise; in skimage, it is count-clockwise
    angle_list_tomopy = ang_list
    angle_list_skimage = -ang_list    
    
    s = rec_bin['trans'].shape # (1, 85, 85)
    # load threshold mask
    mask_bin = m_bin

    # replicate single slice reconstruction to a 3D stack with thickness of "max_thick"
    max_thick = 20 # 30 pixels in height
    mu = {}
    img_cor = {} # absorption corrected by FL.maximium likelihood on binned image, e.g., (1, 85, 85)
    prj_bin = {} # reprojection of "img_cor" on binned image, e.g, (121, 85)
    prj_ratio_bin = {} # ratio of binned image: (prj_bin / sino_bin)
    prj_ratio = {} # resize "prj_ratio_bin" to original sinogram shape, e.g, (121, 680)
    sino_cor = {} # corrected sinogram by: (sino * prj_ratio)
    rec_cor = {} # tomopy mlem reconstruction on corrected sino: (sino_cor)
    img_ratio_bin = {} # ratio of binned image: (img_cor / rec_bin)
    img_ratio = {} # resize of (img_ratio_bin) to original tomogram shape, e.g., (1, 680, 680)
    rec_convert = {} # correced tomography by: (rec * img_ratio)
    sino_smooth = {} # smoothed sino by savgol_filter

    for k in cs.keys():       
        mu[k] = np.ones((max_thick, s[1], s[2])) * cs[k] * rho_2D_bin * mask_bin

    print("calculate attenuation ...")
    attn = {}
    pool = mp.Pool(len(element_list))   
    jobs = []

    for ele in element_list: 
        elem = f'base-{ele}'
        #print(f"calculating attn for {ele} ...")
        jobs.append(pool.map_async(calc_attn, [(ele,mu[elem],mu['base-x'],angle_list_tomopy,s,mask3D,max_thick,pix)]))

    pool.close()
    for job in jobs:
        ele,data = job.get()[0]
        attn[ele] = data
        print(f"got attn data for {ele} ...                    \r", end="")
    pool.join()

    print("calculate img_cor ...")
    img_cor = {}
    pool = mp.Pool(len(element_list))   
    jobs = []
    iter_num = 80 # Ca:200, others: 50

    """
    for ele in element_list: 
        elem = f'base-{ele}'
        jobs.append(pool.map_async(calc_img_cor, 
                                   [(ele,rec_bin[f"xrf_{ele}"],angle_list_skimage,attn,sino_bin,mask_bin,iter_num)]))

    pool.close()
    for job in jobs:
        ele,data = job.get()[0]
        img_cor[f"xrf_{ele}"] = data
        print(f"got img_cor for {ele} ...                  \r", end="")
    pool.join()
    """
    for ele in element_list:
        ele,data = calc_img_cor([ele,rec_bin[f"xrf_{ele}"],angle_list_skimage,attn,sino_bin,mask_bin,iter_num])
        img_cor[f"xrf_{ele}"] = data

    # option 1: 
    # calculate the scaling factor in tomogram, based on binned reconstruction
    for ele in element_list:
        k = f"xrf_{ele}"
        img_ratio_bin[k] = img_cor[k] / rec_bin[k]
        img_ratio[k] = resize(img_ratio_bin[k], s_rec)
        t0 = mf(rec[k][0], 5)
        t1 = mf(img_ratio[k][0], 5)
        rec_convert[k] = np.expand_dims(t0*t1, axis=0)
        rec_convert[k] = rec_convert[k] / binning
    ## end of option 1

    # option 2:
    # calculate the scaling factor in sinogram, then reconstruct the corrected sinogram using tomopy
    ## reprojection
    pool = mp.Pool(len(element_list))   
    jobs = []
    iter_num = 80 # Ca:200, others: 50

    n_angle = len(angle_list_skimage)
    for ele in element_list:
        print(f"processing {ele} ...    \r", end="")
        k = f"xrf_{ele}"
        s_cor = img_cor[k].shape
        prj_bin[k] = np.zeros(( n_angle, s[-1]))
        for i in range(n_angle):
            tmp = rot3D(img_cor[k][0], angle_list_skimage[i])
            prj_bin[k][i] = np.sum(tmp, axis=0, keepdims=True)

        r = prj_bin[k] / sino_bin[k][0]
        r[np.isnan(r)] = 1
        r[np.isinf(r)] = 1
        prj_ratio_bin[k] = r
        #prj_ratio[k] = resize(prj_ratio_bin[k], (n_angle, int(s_cor[-1]*binning)))
        prj_ratio[k] = resize(prj_ratio_bin[k], sino[k].shape[1:])
        prj_ratio[k][np.isnan(prj_ratio[k])] = 0
        prj_ratio[k][np.isinf(prj_ratio[k])] = 0
        prj_ratio[k] = np.expand_dims(prj_ratio[k], axis=0)
        sino_smooth[k] = sino[k].copy() # (121, 680)
        sino_cor[k] = sino_smooth[k] * prj_ratio[k]

        theta_list = angle_list_tomopy /180.0 * np.pi
        rc = s_cor[-1] * binning / 2- 0.5
        jobs.append(pool.map_async(recon_mlem, [(k, sino_cor[k][0], theta_list, rc, iter_num)]))

    pool.close()
    for job in jobs:
        k,data = job.get()[0]
        rec_cor[k] = data
        print(f"got rec_cor for {k} ...       ")
    pool.join()
    ## end of option 2
    
    for ele in element_list:
        k = f"xrf_{ele}"
        if isinstance(dt.proc_data['overall']['maps'][k], list):        
            mms = dt.proc_data['overall']['maps'][k][0].copy()
            mmt = dt.proc_data['overall']['tomo'][k][0].copy()
        else:
            mms = dt.proc_data['overall']['maps'][k].copy()
            mmt = dt.proc_data['overall']['tomo'][k].copy()
        mms.d = sino_cor[k][0]
        dt.proc_data['overall']['maps'][f'{k}_cor'] = mms
        mmt.d = rec_convert[k][0]
        dt.proc_data['overall']['tomo'][f'{k}_cor'] = mmt
        
    dt.save_data(save_sns='overall', save_data_keys=['maps', 'tomo'], save_sub_keys=[f"xrf_{ele}_cor" for ele in element_list])