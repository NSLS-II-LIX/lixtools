from lixtools.hdf import h5xs_an
from py4xs.hdf import lsh5,h5xs,proc_merge1d,h5_file_access
from py4xs.data2d import Data2d,MatrixWithCoords,unflip_array
from py4xs.slnxs import Data1d

import h5py
import numpy as np
import cv2

#Class to handle fiber diffraction data from LIX. 
#To do: replace methods here with those from py4xs MatrixWithCoordinates.
class h5xs_fiber(h5xs_an):

    def __init__(self, fn,*args, **kwargs):
        super().__init__(fn,*args, **kwargs)

    def proc_muscle(self,fns,params):
        """
        standard muscle processing pipeline given params provided by user.
        
        Process features. Iterates through a list of dicts with inputs for feature detection. 
        feature = {
        'label': "M3"; whichever label that makes sense to the user
        'direction': 'x'; for line analysis through the radial direction and averaging along the azimuthal direction
        'direction': 'y'; for line analysis through the azimuthal direction and averaging along the radial direction
        'qx_min': some number greater than zero
        'qx_max': some number greater than qx_min
        'qy_min': some number greater than zero
        'qy_max': some number gerater than qy_min
        }        
        """

        return None

    def proc_muscle_livestreaming(self,sn,params):
        """
        standard muscle processing pipeline for live streaming data using preset parameters.
        Intended for quick checking of data and not of final processed data.
        """

        self.process_qxy_fiber(sn=sn,subpixel = params.subpixel,mask=params.mask)
        qxy_list = self.proc_data[sn]['processed']['qxy_SAXS'].copy() #list of data2s
        N = len(qxy_list)

        dd2 = qxy_list[0].copy()
        lines_equator = [None]*N
        lines_meridionals = [None]*N
        for i,qxy in enumerate(qxy_list):
            _,lines_equator[i]    = self.grab_fiber_line_profile(data2d = qxy,direction = 'x',qx_min = 0,qx_max = 1e2,
                                                                                              qy_min = 0,qy_max = 0.005 * 2*3.14159/10)
            _,lines_meridionals[i] = self.grab_fiber_line_profile(data2d = qxy,direction = 'y',qx_min = 0,qx_max = 0.005 * 2*3.14159/10,
                                                                                               qy_min = 0,qy_max = 1e2)
        dd2.d = np.array(lines_equator)
        self.add_proc_data(sn, 'processed', 'equators',dd2)
        dd2 = dd2.copy() 
        dd2.d = np.array(lines_meridionals)
        self.add_proc_data(sn, 'processed', 'meridionals',dd2)
        self.save_data([sn], ['processed'])
        
        #Blind merging for quick checks
        self.Merge_qxy(sn)
        return None

    @h5_file_access
    def process_qxy_fiber(self,sn,subpixel=True,mask = None,debug=True):
        """ produce qx-qy maps corresponding to each exposure for fiber data
        """
        det_ext = "_SAXS"
        det = self.detectors[0]
        map_key = det_ext

        exp = det.exp_para
        cx = exp.bm_ctr_x
        cy = exp.bm_ctr_y
        dr = exp.ratioDw*exp.ImageWidth   # detector distance in pixels 
        dq = 4.*np.pi/exp.wavelength*np.sin(0.5/dr)
        if mask is None: #User can give their own mask (e.g. ones with imdilation/imerosion if desired)
            mask = exp.mask.map
        else:
            assert mask.shape == exp.mask.map.shape, "Mask input must be of shape ({},{})".format(exp.mask.map.shape[0],exp.mask.map.shape[1])
            mask = mask>0 #Convert to boolean for indexing
        negativemask = (~mask).astype(np.uint8)
        negativemask = CenterByPadding(negativemask,cy,cx,subpixel=subpixel)
        if subpixel:
            #subpixel interpoalation 'bleeds' into neighboring pixel at the boundary of a mask
            kernel = np.ones((3,3),dtype = np.uint8)
            negativemask = cv2.erode(negativemask,kernel)

        results = {}
        
        if debug:
            print(f"processing {sn} ...")
        dh5 = self.h5xs[sn]
        with h5py.File(dh5.fn, "r") as fh:
            imgs = np.array(fh[sn]["pil/data/pil1M_image"],dtype = np.float32)
            imgs = np.rot90(imgs, exp.flip, [-2, -1])

            #Center by padding and rotate
            ret = []
            fiberphi = []
            for i in range(len(imgs)):
                #Auto align to fiber axis
                im = CenterByPadding(imgs[i],cy,cx,subpixel=subpixel)
                im,rotated_negativemask,phi = Align(im*negativemask,negativemask)
                
                #Quadrant fold with gap removal
                wts = (rotated_negativemask + np.fliplr(rotated_negativemask) 
                    + np.flipud(rotated_negativemask) + np.fliplr(np.flipud(rotated_negativemask))).astype(np.float32)
                wts[wts==0] = np.inf #Sends edge case of division by zero to zero.

                im = im + np.fliplr(im) + np.flipud(im) + np.fliplr(np.flipud(im))
                im /= wts
                ret.append(im)
                fiberphi.append(phi*180/np.pi)
        results[sn] = [np.array(ret),fiberphi]


        for sn,pack in results.items():  
            #unpack
            arr_data = pack[0]
            fiberphi = pack[1]

            #Only save one quadrant since they are all identical
            (l,m,n) = arr_data.shape
            x = np.arange(m//2+1) 
            y = np.arange(n//2+1) 

            dd2 = MatrixWithCoords()
            dd2.xc = dq*x
            dd2.xc_label = "qx"
            dd2.yc = dq*y
            dd2.yc_label = "qy"
            dd2.err = None
            maps = []

            for i in range(l):
                dd2.d = arr_data[i,m//2:,n//2:].T 
                maps.append(dd2.copy())

            phis = dd2.copy()
            phis.d = np.array(fiberphi)  
            self.add_proc_data(sn, 'processed', 'qxy'+map_key, maps)
            self.add_proc_data(sn, 'processed', 'fiber_phi(deg)', phis)
            self.save_data([sn], ['processed'])

    def grab_fiber_line_profile(self,data2d,direction,qx_min,qx_max,qy_min,qy_max,**kwargs):
        """
        takes in a py4xs/data2d object generated from process_qzy_fiber or Merge_qxy so there's a presumed data structure
        i.e. data2d.d is the +/+ quadrant of the fiber image
        data2d.yc is the azimuthal direction with range [0,yc.max()]
        data2d.xc is the radial direction with range [0,xc.max()] 
        """
        assert qx_min>=0, "qx_min must be greater than or equal to zero"
        assert qx_max>qx_min, "qx_max must be greater than qx_min"
        assert qy_min>=0, "qy_min must be greater than or equal to zero"
        assert qy_max>qy_min, "qy_max must be greater than qy_min"
        assert direction in ['x','y'], "direction must be either 'x' for fiber radial direction or 'y' for fiber azimuthal direction"

        #py4xs data2d.roi used in line_profile flips the yc for some reason. Ok....
        #so set yrange to [yc.max()-qmax,yc.max()] instead of [0,qmax]

        if direction == 'x':
            # line_q,line_I,_ = data2d.line_profile(direction= 'x', xrange=None, yrange=[data2d.yc.max()-(qy_max-qy_min),data2d.yc.max()-qy_min])
            line_q,line_I,_ = data2d.line_profile(direction= 'x', xrange=None, yrange=[data2d.yc.max()-qy_max,data2d.yc.max()])
        elif direction == 'y':
            line_q,line_I,_ = data2d.line_profile(direction= 'y', yrange=None, xrange=[qx_min,qx_max])
        return line_q,line_I


    def Merge_qxy(self,sn,removal_index = None):
        qxy = self.proc_data[sn]['processed']['qxy_SAXS'].copy() #list of data2s
        N = len(qxy)
        
        #Remove from list
        if removal_index is not None:
            removal_index = sorted(removal_index,reverse = True)
            [qxy.pop(i) for i in removal_index]
        merged = MatrixWithCoords()
        merged.xc_label = "qx"
        merged.xc = qxy[0].xc
        merged.yc_label = "qy"
        merged.yc = qxy[0].yc
        merged.d = np.zeros_like(qxy[0].d)
        merged = merged.merge(qxy)
        
        #Prettify the merge by showing all four quadrants. Even though they are identical.
        n,m = merged.d.shape
        new = np.zeros((2*n-1,2*m-1))
        new[:n,:m] = np.flipud(np.fliplr(merged.d))
        new[:n,m:] = np.flipud(merged.d[:,:-1])
        new[n:,:m] = np.fliplr(merged.d[:-1,:])
        new[n:,m:] = merged.d[:-1,:-1]
        merged.d = new
        self.add_proc_data(sn, 'processed','merged_qxy_SAXS' , merged)
        self.save_data([sn], ['processed'],['merged_qxy_SAXS'])
        self.set_h5_attr(f"{sn}/processed/{'merged_qxy_SAXS'}", "Nimages", len(qxy))

  

def CenterByPadding(image,centeri,centerj,subpixel=True,interp='Default'):
    #Enforces NxM where N and M odd. So N//2 and M//2 is always at center of image

    #numpy ij indexing
    l = image.shape[0]
    m = image.shape[1]

    #Move to center by zeropadding image
    lm1 = l-1
    int_centeri = int(centeri)
    int_centerj = int(centerj)
    if int_centeri >= l//2:
        pad0 = (0,2*int_centeri-lm1)
    elif centeri < l//2:
        pad0 = (lm1-2*int_centeri,0)

    mm1 = m-1
    if int_centerj >= m//2:
        pad1 = (0,2*int_centerj-mm1)
    elif int_centerj < m//2:
        pad1 = (mm1-2*int_centerj,0)
    image = np.pad(image, (pad0,pad1)) 

    if subpixel:
        #Subpixel shift to corner of pixel 
        if interp == 'Default':
            interp = cv2.INTER_LINEAR
        # image = np.pad(image, ( (1,1),(1,1))) #pad all sides by 1 pixel to avoid info loss  #Just doesn't matter enough for large images
        l2 = image.shape[0]
        m2 = image.shape[1]

        ishift = centeri % 1
        jshift = centerj % 1
        # cv2 translation matrix
        translation_matrix = np.array([
            [1, 0, -jshift],
            [0, 1, -ishift]
        ], dtype=np.float32)
        image = cv2.warpAffine(src=image,
                M=translation_matrix,
                dsize=(m2, l2),
                flags = interp)
    return image

def rotate_image(image,center, angle):
    #cv2 xy indexing
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def Align(image,negativemask,phi = None,align_threshold = 5):
    """ Orientate images based on image moments. NB: cv2's xy is flipped from numpy's ij
    """
    if phi is None:
        imagethres = negativemask*(image>align_threshold)
        imagethres = imagethres.astype(np.float32)
        moments = cv2.moments(imagethres) 
        if moments['mu20'] == moments['mu02']:
            phi = 0
        else:
            phi = np.arctan(2*moments['mu11']/(moments['mu20']-moments['mu02']))/2
        w,v = np.linalg.eig(np.array([ [moments['mu20'],moments['mu11']] , [moments['mu11'],moments['mu02']]  ]))        
       
        i = abs(w).argmax()
        if abs(v[i,0])>abs(v[i,1]):
            phi = np.pi/2 + phi
    #Assumes rotation center is center of image.
    centerx = image.shape[1]//2
    centery = image.shape[0]//2
    image = rotate_image(image.astype(np.float32),(centerx,centery),phi*180/np.pi)
    negativemask = rotate_image(negativemask.astype(np.uint8),(centerx,centery),phi*180/np.pi) 
    return image,negativemask,phi     


