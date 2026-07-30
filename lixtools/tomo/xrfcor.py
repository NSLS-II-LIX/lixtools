# based on forward calculation implemented by Mani Valleti
# manuscript in prepration
# assited by ChatGPT

import tensorflow as tf
import numpy as np
import xraydb
from .tfutils import tfrotate
from py4xs.data2d import MatrixWithCoords
from lixtools.hdf import h5xs_scan
from tomopy.recon.rotation import find_center_vo
from scipy.ndimage import shift

def abs_len_Edep(e, e0=15126.4):
    """ approximate energy dependence of the basorption length as a function of energy
    """
    # simplest form of approximation; absorption length obeys a power law
    #eratio = e/e0
    #return np.power(eratio, 3.03)
    
    # a bit more accurate; fitting data using a polynomial
    p = np.poly1d([-2.11713498e-02,  8.71124729e-01, -1.43569679e+01,  
                   1.18583491e+02, -4.88337577e+02,  7.88831304e+02])
    return np.exp(p(np.log(eratio*e0)))    

def abs_len_Edep_tf(e):
    e0=15126.4
    p = np.poly1d([-2.11713498e-02,  8.71124729e-01, -1.43569679e+01,  
                   1.18583491e+02, -4.88337577e+02,  7.88831304e+02])
    return tf.math.exp(p(tf.math.log(e)))    

def get_incattn(abs_map, angles, pix = 1.):
    """
    :param abs_map: tf.tensor [B = 1, H, W, C = 1], reconstructed absorption map, 
    :param angles: tf.tensor, tensforflow angles object in radians
    :param pix: float, pixel size
    """

    pixtf = tf.constant(pix, dtype = tf.float32)
    nang = angles.shape[0]
    inc_tp = tf.transpose(abs_map, [3,1,2,0])      # [C=1, H, W, B = 1]
    inc_tiled = tf.tile(inc_tp, [nang,1,1,1])

    # This is the absorption map rotated using the angles specified
    inc_rot = tfrotate(inc_tiled, angles, interpolation = "bilinear")      # [Nang, H, W, B = 1]
    inc_rot = tf.transpose(inc_rot, [3,0,1,2])      # [B = 1, Nang, H, W]

    inc_mudl = inc_rot*pixtf     # The attenuation i.e., mu.dl per pixel for every pixel at every rotation angle
    tau_inc = tf.cumsum(inc_mudl, axis = -1, exclusive=True)     # sum(mu.dl), [B = 1, Nang, H, W]
    tau_inc = tf.clip_by_value(tau_inc, 0.0, 250.0)      #Numerical stability

    A_in = tf.exp(-tau_inc)    # exp(-sum(mu.dl))  [B = 1, Nang, H, W]

    A_in = A_in[0,...,None]     # [Nang, H, W, C = 1], consistency so A_in and P_out shapes stay the same

    return A_in


def get_emsattn(abs_map, angles, krnl, inc_ene, ems_ene, pix = 1.):
    """
    :param abs_map: tf.tensor [B = 1, H, W, C = 1], reconstructed absorption map, 
    :param angles: tf.tensor, tensforflow angles object in radians
    :param krnl: tf.tensor [kH, kW, C_in, C_out], kernel describing the emission path for each pixel
    :param e_ratio: float, (inc_ene/ems_ene)
    :param pix: float, pixel size
    """
    pixtf = tf.constant(pix, dtype = tf.float32)
    nang = angles.shape[0]

    #e_ratio = inc_ene/ems_ene
    #e_ratiotf = tf.constant(e_ratio, dtype = tf.float32)
    #absmap_emsenergy = abs_map*(tf.math.pow(e_ratiotf, 3.03))
    abs_len_ratio = tf.cast(abs_len_Edep_tf(inc_ene)/abs_len_Edep_tf(ems_ene), dtype = tf.float32)
    absmap_emsenergy = abs_map*abs_len_ratio
    
    # Rotating the absorption map
    ems_tp = tf.transpose(absmap_emsenergy, [3,1,2,0])      # [C=1, H, W, B = 1]
    ems_tiled = tf.tile(ems_tp, [nang,1,1,1])

    # This is the emission absorption map rotated using the angles specified
    ems_rot = tfrotate(ems_tiled, angles, interpolation = "bilinear")      # [Nang, H, W, B = 1]
    ems_rot = tf.transpose(ems_rot, [3,0,1,2])      # [B = 1, Nang, H, W]

    ems_mudl = ems_rot*pixtf     # The attenuation i.e., mu.dl per pixel for every pixel at every rotation angle
    taus = []

    for i in range(nang):
        taus.append(tf.nn.conv2d(ems_mudl[0, i:i+1,...,None],       # One angle slice at a time
                                tf.cast(krnl, dtype = tf.float32), 
                                strides=1, padding='SAME'))

    tau_ems = tf.concat(taus, axis = 0)     # [Nang, H, W, C = 1]

    tau_ems = tf.clip_by_value(tau_ems, 0, 200)
    P_out = tf.exp(-tau_ems)        # [Nang, H, W, C = 1]

    return P_out        # [Nang, H, W, C = 1]

def build_emission_kernel(
    H_img=661, W_img=661,
    det_distance_mm=60.0,        # sample → detector distance (mm)
    det_area_mm2=64.0,           # assume ~8x8 mm detector
    kH=None, kW=None,            # optional overrides; if None, computed from geometry
    row_normalize=False,         # False = pure path-length style; True = per-depth normalization
    downstream = True,           # True = detector below, False = detector above
    dtype=np.float32
):
    # # half-angle from geometry (square detector assumed: side = sqrt(area))
    # side_mm = np.sqrt(det_area_mm2)
    # half_side_mm = side_mm * 0.5
    # alpha = np.arctan(half_side_mm / det_distance_mm)   # radians

    # half-angle from geometry (circular detector assumed)
    solid_angle = det_area_mm2/(det_distance_mm**2)
    alpha = np.arccos(1-(solid_angle/(2*np.pi)))

    
    # kernel height: make odd and ≥ 2*H_img-1 so center sits at emitter plane
    if kH is None:
        kH = 2*H_img + 1   # e.g., 1323 when H_img=661
    if kH % 2 == 0:
        kH += 1
    cH = kH // 2

    # kernel lateral width from max depth (farthest pixel → ~H_img)
    # halfwidth_pixels ≈ tan(alpha) * max_dy_pixels
    if kW is None:
        halfw = int(np.ceil(np.tan(alpha) * W_img))
        kW = 2*halfw + 1
    if kW % 2 == 0:
        kW += 1
    cW = kW // 2

    K = np.zeros((kH, kW), dtype=dtype)

    # fill only the downstream half (rows >= cH)
    for r in range(cH, kH):
        dy = r - cH                      # pixels "downstream" toward detector
        if dy == 0:
            # only the central column contributes at the emitter plane
            K[r, cW] = 1.0
            continue

        # lateral half-width allowed by cone at this depth
        halfw_dy = int(np.floor(np.tan(alpha) * dy))
        left = max(cW - halfw_dy, 0)
        right = min(cW + halfw_dy, kW - 1)

        # fill secant weights across lateral support
        offs = np.arange(left, right+1) - cW
        # sec = 1/cos(arctan(|dx|/dy)) = sqrt(1 + (dx/dy)^2)
        sec = np.sqrt(1.0 + (offs.astype(np.float64) / float(dy))**2)
        K[r, left:right+1] = sec.astype(dtype)

        if row_normalize:
            s = K[r, :].sum()
            if s > 0:
                K[r, :] /= s
    
    # === NEW BEHAVIOR: flip vertically if detector is ABOVE the sample ===
    if not downstream:
        K = np.flipud(K).copy()

    # shape for conv2d: [kH, kW, C_in, C_out] with C_in=C_out=1
    K4 = K[:, :, None, None]
    return K4, alpha

def shift_sino_center_integer(sino, center):
    """ Shifts sinogram wrt to the Center of Rotation
        After perfomring this operation, the image could be treated as if COR coincides with
        the geometric center
    """
    dx = sino.shape[-1]
    delta = center - dx/2.0
    return np.roll(sino, shift=int(round(-delta)), axis=-1)
        
def shift_sino_center(sino, center):
    dx = sino.shape[-1]
    delta = center - dx/2.0
    return shift(sino, [0, -delta])
    
def get_emission_line_energy(line):
    """ line should include element and edge, e.g. "Fe_K"
    """
    ele,edge = line.split("_") 
    # this returns all lines, e.g. 
    #   'Ka3': XrayLine(energy=6267.4, intensity=0.000276096, initial_level='K', final_level='L1'),
    #   'Ka2': XrayLine(energy=6392.1, intensity=0.294023, initial_level='K', final_level='L2'),
    #   'Ka1': XrayLine(energy=6405.2, intensity=0.580277, initial_level='K', final_level='L3'),
    #   'Kb3': XrayLine(energy=7059.3, intensity=0.0425566, initial_level='K', final_level='M2'),
    #   'Kb1': XrayLine(energy=7059.3, intensity=0.0821556, initial_level='K', final_level='M3'),
    #   'Kb5': XrayLine(energy=7110.0, intensity=0.000712115, initial_level='K', final_level='M4,5')}
    xls = xraydb.xray_lines(ele, edge)

    # return weighted average
    e_em = np.sum([l.energy*l.intensity for l in xls.values()])/np.sum([l.intensity for l in xls.values()])
    # alternatively, return the brightest line 

    return e_em

class XRFmlem:
    def __init__(self, dt: h5xs_scan, n_iter=20, reduce_mode='sum'):
        """ LiX data 
            angles in degrees
        """
        #angles, img_width, incident_energy
        dt.load_data("overall", ['maps', 'tomo'])
        if not 'absorption' in dt.proc_data['overall']['tomo'].keys(): 
            raise Exception("Can't find the absorption tomogram!")
        img_width = len(dt.proc_data['overall']['maps']['absorption'].xc)
        angles = dt.proc_data['overall']['maps']['absorption'].yc
        incident_energy = dt.h5xs[dt.samples[0]].header()['energy']['energy']

        self.dt = dt
        self.rec = None
        self.img_width = img_width
        self.ang = np.deg2rad(270-angles)
        self.image_shape = (img_width, img_width)
        self.sino_shape = (len(angles), img_width)
        
        self.e_inc = incident_energy
        self.mode = reduce_mode
        self.Ain = None
        self.Pout = None

        self.abs_tomo = self.recon("absorption", n_iter=n_iter)

    def calc_abs(self, emission_line):
        """ abs_map is the measured absorption tomogram
            but must be formated match 
        """
        #abs_tomo = self.dt.proc_data['overall']['tomo']['absorption'].d
        abs_tomo_tf = tf.cast(self.abs_tomo[None,...,None], dtype = tf.float32)
        self.Ain = get_incattn(abs_tomo_tf, self.ang)

        K4_dn, _ = build_emission_kernel(self.img_width, self.img_width, row_normalize=True)     # Downstream Kernel
        K4_up, _ = build_emission_kernel(self.img_width, self.img_width, row_normalize=True, downstream = False)   # Upstream Kernel
        #eratio = self.e_inc/get_emission_line_energy(emission_line)
        #Pout_dn = get_emsattn(abs_map, self.ang, K4_dn, e_ratio=eratio)
        #Pout_up = get_emsattn(abs_map, self.ang, K4_up, e_ratio=eratio)
        ems_ene = get_emission_line_energy(emission_line)
        Pout_dn = get_emsattn(abs_tomo_tf, self.ang, K4_dn, self.e_inc, ems_ene)
        Pout_up = get_emsattn(abs_tomo_tf, self.ang, K4_up, self.e_inc, ems_ene)
        self.Pout = tf.concat([Pout_dn, Pout_up], axis = 3)        
        
    def forward(self, rec):
        nang = self.ang.shape[0]
        img = tf.transpose(rec, [3, 1, 2, 0])
        img = tf.tile(img, [nang, 1, 1, 1])
        img = tfrotate(img, self.ang, interpolation="bilinear")
    
        if self.Ain is not None:
            img = tf.math.multiply(img, self.Ain)
        
        if self.Pout is not None:
            img = tf.math.multiply(img, self.Pout)
    
        if self.mode == 'mean':
            sino = tf.reduce_mean(img, 2, name=None)
        elif self.mode == 'sum':
            sino = tf.reduce_sum(img, 2, name=None)
        
        if sino.shape[-1]==2:
            # Multi Channel scenario, we just collapse the dimension here
            ch0 = sino[..., 0]                           # (Nang, H)
            ch1 = tf.reverse(sino[..., 1], axis=[1])     # flip spatial dim
    
            sino = tf.concat([ch0, ch1], axis=0)[..., None]        # (2*Nang, H, 1)
        
        sino = tf.reshape(sino, [1, sino.shape[0], sino.shape[1], sino.shape[2]])   # (1, Nang, H, 1)
    
        return sino

    def backproject(self, sino):
        """
        Computes A^T sino using TensorFlow autodiff.
        """
    
        x = tf.ones((1,self.img_width,self.img_width,1), dtype=tf.float32)
    
        with tf.GradientTape() as tape:
            tape.watch(x)
            proj = self.forward(x)
            inner = tf.reduce_sum(proj * sino)
    
        backproj = tape.gradient(inner, x)
    
        return backproj

    def compute_sensitivity(self):
    
        if self.Pout is None:
            ones_sino = tf.ones(self.sino_shape, dtype=tf.float32)
        else:
            ones_sino = tf.ones((len(self.ang)*2, self.img_width), dtype=tf.float32)

        sens = self.backproject(ones_sino[None,:,:,None])
    
        return sens
    
    
    def recon(self, map_key, n_iter=10, eps=1e-8):
        """ XRF sinograms should have the shape (2, N_angle, img_width)
            corresponding to the detector on both sides of the sample
            the sinograms should be shifted such that the rotation center is mid-image
        """

        try:
            mm = self.dt.proc_data['overall']['maps'][map_key]
        except:
            raise Exception(f"unable to load sinogram for {map_key} ...")

        cen = self.dt.get_h5_attr("overall/maps", "rot_cen")   # this should have been set already

        if isinstance(mm, MatrixWithCoords):
            measured_sino = mm.d
        elif len(mm)==4:  # LiX 4-channel detector
            xrf_ch0 = (mm[0].d+mm[1].d)/2
            xrf_ch1 = (mm[2].d+mm[3].d)/2
            measured_sino = np.stack([xrf_ch0, xrf_ch1])
        else:
            raise Exception(f"currently cannot deal with XRF detector with {len(mm)} channels")
            
        assert self.sino_shape == measured_sino.shape[-2:]
        if cen:
            measured_sino = shift_sino_center_integer(measured_sino, cen)
        if len(measured_sino.shape)==3: # XRF      
            measured_sino = np.concatenate([measured_sino[0], np.fliplr(measured_sino[1])], axis=0)
        
        # Initial estimate
        x = tf.ones((1,self.img_width,self.img_width,1), dtype=tf.float32)
    
        sensitivity = self.compute_sensitivity()
        sensitivity = tf.maximum(sensitivity, eps)
    
        for k in range(n_iter):
    
            # Forward projection
            estimated_sino = self.forward(x)
            estimated_sino = tf.maximum(estimated_sino, eps)
    
            # Ratio term
            ratio = measured_sino[None,:,:,None] / estimated_sino
    
            # Backprojection of ratio
            correction = self.backproject(ratio)
    
            # EM update
            x = x * correction / sensitivity
    
            # enforce positivity
            x = tf.maximum(x, 0.0)

        self.rec = x[0,:,:,0]
        return self.rec

