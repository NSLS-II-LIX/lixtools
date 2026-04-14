from lixtools.hdf import h5xs_an
from py4xs.hdf import lsh5,h5xs,proc_merge1d,h5_file_access
from py4xs.data2d import Data2d,MatrixWithCoords,unflip_array
from py4xs.slnxs import Data1d
import json
import h5py
import numpy as np
import cv2

from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from lixtools.fiber import MuscleDiffraction

c = 2*3.14159/10#1/nm to 1/angstroms

#Class to handle fiber diffraction data from LIX. 
#To do: replace methods here with those from py4xs MatrixWithCoordinates.
class h5xs_fiber(h5xs_an):

    def __init__(self, fn,*args, **kwargs):
        super().__init__(fn,*args, **kwargs)

    def proc_box(self,sns,box):
        """
        Process features given by box parameters. Iterates through a list of dicts with inputs for feature detection. 
        box = {
        'label': "M3"; whichever label that makes sense to the user
        'direction': 'x'; for line analysis through the radial direction and averaging along the azimuthal direction
        'direction': 'y'; for line analysis through the azimuthal direction and averaging along the radial direction
        'qx_min': number greater than zero in absolute units (1/angstrom)
        'qx_max': some number greater than qx_min
        'qy_min': number greater than zero in absolute units (1/angstrom)
        'qy_max': some number gerater than qy_min
        'PrincipalSpacing': A principal spacing in nm that's meaningful to user. 
        'peaks': dict with 4 keys to fit gaussians
                'relative_qmin' and 'relative_qmax': min and max qvalue relative to 1/PrincipalSpacing*2*pi/10 to bound the gaussian fits 
                'absolute_smin' and 'absolute_smax': min and max sigmas in absolute values (1/angstrom) to bound gaussian fits
        'update_keys': list of list of peak keys to fit multiple gaussians at once. Useful when gaussians overlap. i.e. [ ['10' ,'11']] or [['M2_subpeakA','M2_subpeakB']] to jointly fit the peaks.
        }        
        """

        self.load_data(sns,read_data_keys=['processed'],read_sub_keys=['qxy_SAXS'])

        #Equator fitting
        LineDatas = []
        for sn in sns:
            qxy_list = self.proc_data[sn]['processed']['qxy_SAXS'].copy() 
            for qxy in qxy_list:
                LineDatas.append( self.fit_line_profile(qxy,box) )
        self.add_proc_data('overall',box['label'],'q(angstrom-1)',LineDatas[0].q)
        self.add_proc_data('overall',box['label'],'d(nm)',c/LineDatas[0].q)
        self.add_proc_data('overall',box['label'],'values'     ,np.array( [LineData.values          for LineData in LineDatas]    ))
        self.add_proc_data('overall',box['label'],'backgrounds',np.array( [LineData.background      for LineData in LineDatas]    ))
        self.add_proc_data('overall',box['label'],'signals'    ,np.array( [LineData.filtered_values for LineData in LineDatas]    ))
        self.add_proc_data('overall',box['label'],'fits'       ,np.array( [LineData.fitted_values   for LineData in LineDatas]    ))
        if box['label'] in ['equator','Equator','equators','Equators']:
            self.add_proc_data('overall',box['label'],'IR'         ,np.array( [LineData.peaks['11']['Area']/(LineData.peaks['10']['Area']+1e-9) 
                                                                                                                for LineData in LineDatas]    ))
        for key in box['peaks'].keys():
            self.add_proc_data('overall',box['label'],key + '/dspacing(nm)'     ,np.array( [c/LineData.peaks[key]['m1']
                                                                                                                for LineData in LineDatas]    ))
            self.add_proc_data('overall',box['label'],key + '/area(count nm-1)' ,np.array( [LineData.peaks[key]['Area']/c
                                                                                                                for LineData in LineDatas]    ))
            self.add_proc_data('overall',box['label'],key + '/sigma(nm-1)'      ,np.array( [LineData.peaks[key]['m2']/c
                                                                                                                for LineData in LineDatas]    ))
        self.save_data(['overall'],[box['label']])
        self.set_h5_attr(f"overall/{box['label']}",'box_params', json.dumps(box))


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

        new = data2d.copy()
        new.yc = np.flip(new.yc) #hot fix. There's an origin shift SOMEWHERE
        line_q,line_I,_ = new.line_profile(direction= direction, xrange=[qx_min,qx_max], yrange=[qy_min,qy_max])
        if direction == 'y':
            line_q = np.flip(line_q)#hot fix. There's an origin shift SOMEWHERE
        return line_q,line_I


    def fit_line_profile(self,qxy,box_params,backgroundRemoval = True):
        d0 = box_params['PrincipalSpacing']
        q0 = (1/d0)*c
        LineData = MuscleDiffraction.MuscleLineData(q = None, y = None) 
        LineData.q , LineData.values = self.grab_fiber_line_profile(data2d = qxy,
                                                              direction = box_params['direction'],
                                                              qx_min = box_params['qx_min'],
                                                              qx_max = box_params['qx_max'],
                                                              qy_min = box_params['qy_min'],
                                                              qy_max = box_params['qy_max'],)
        if backgroundRemoval:
            LineData.BackgroundRemoval(monotonic(LineData.q,LineData.values))
        else: #Background is already removed
            LineData.filtered_values = LineData.values
            LineData.background = np.zeros(LineData.values.shape)
        for key in box_params['peaks'].keys():
            peak = box_params['peaks'][key]
            bounds = [(0,1),(q0*peak['relative_qmin'],q0*peak['relative_qmax']),(peak['absolute_smin'],peak['absolute_smax']) ] #bounds on single gaussian fit
            LineData.FitSingleGaussian(LineData.q,LineData.filtered_values,label = key,maxiter = 1000,bounds = bounds) #initial fits
            LineData.peaks[key]['smin'] = peak['absolute_smin']
            LineData.peaks[key]['smax'] = peak['absolute_smax']
        for update_keys in box_params['update_keys']:
            if box_params['update_method'] == 'NGaussian':
                LineData.NGaussianFitKeys(update_keys,maxiter=1000) #Fit 10 and 11 together using initial fits as guesses
            elif box_params['update_method'] == 'NGaussianCluster':
                LineData.FitClusterWithGaussians(update_keys,maxiter=1000)
        LineData.ComputeFittedValues(box_params['peaks'].keys())

        return LineData

    def fit_line_profile_M3_radial(self,qxy,M3_box_params):#ad foc function to handle M3 radial. Need to refactor code to generalize this:

        #custom background subtraction
        new = qxy.copy()
        n = new.d.shape[1]
        LineData = MuscleDiffraction.MuscleLineData(q = None, y = None) 
        LineData.q = new.yc
        for i in range(n):
            LineData.values = new.d[:,i]
            LineData.values[:100] = LineData.values.max()
            LineData.BackgroundRemoval(monotonic(LineData.q,LineData.values))
            new.d[:,i] = LineData.filtered_values

        return self.fit_line_profile(new,M3_box_params,backgroundRemoval = False)

    def proc_muscle_merge(self,sns,params):
        merge_settings = params.merge_settings

        qxy_list = []
        em2_list = []
        for sn in sns:
            qxy_list += self.proc_data[sn]['processed']['qxy_SAXS']
            em2_list += list(self.fh5[sn]["em2/data/em2_ts_SumAll"])
        removal_bool = [False]*len(qxy_list)

        #Exclusion criteria
        IR  = self.proc_data['overall'][params.equator_box['label']]['IR']
        removal_bool = np.logical_or(removal_bool,  np.logical_or(IR<merge_settings['IRmin'],IR>merge_settings['IRmax'])    )
        d10 = self.proc_data['overall'][params.equator_box['label']]['10/dspacing(nm)']
        removal_bool = np.logical_or(removal_bool,  np.logical_or(d10<merge_settings['d10min'],d10>merge_settings['d10max'])    )
        A10 = self.proc_data['overall'][params.equator_box['label']]['10/area(count nm-1)']
        removal_bool = np.logical_or(removal_bool,  np.logical_or(A10<merge_settings['Area10min'],A10>merge_settings['Area10max'])    )
        
        removal_index = np.arange(len(qxy_list))[removal_bool]

        if len(removal_index) == len(qxy_list):
            print('No images left after exclusion criteria. Now blindly merging all images')
            removal_index = []

        #Remove from list
        removal_index = sorted(removal_index,reverse = True)
        for index in removal_index:
            qxy_list.pop(index)
            em2_list.pop(index)

        merged = self.Merge_qxy(qxy_list)

        for box in params.boxes:
            if box['label'] == 'M3_radial':
                LineData = self.fit_line_profile_M3_radial(merged,box) #ad hoc function for now. To be generalized
            else:
                LineData = self.fit_line_profile(merged,box)
            self.add_proc_data('overall','merged',box['label']+'/values'      ,LineData.values)
            self.add_proc_data('overall','merged',box['label']+'/backgrounds' ,LineData.background)
            self.add_proc_data('overall','merged',box['label']+'/signals'     ,LineData.filtered_values)
            self.add_proc_data('overall','merged',box['label']+'/fits'        ,LineData.fitted_values)
            self.add_proc_data('overall','merged',box['label']+'/q(angstrom-1)',LineData.q)
            self.add_proc_data('overall','merged',box['label']+'/d(nm)',c/LineData.q)
            for key in box['peaks'].keys():
                self.add_proc_data('overall','merged',box['label']+'/' + key + '/' + 'dspacing(nm)', np.array(c/LineData.peaks[key]['m1']))
                self.add_proc_data('overall','merged',box['label']+'/' + key + '/' + 'sigma(nm-1)', np.array(LineData.peaks[key]['m2']/c))
                self.add_proc_data('overall','merged',box['label']+'/' + key + '/' + 'area(count nm-1)', np.array(LineData.peaks[key]['Area']/c))


        self.add_proc_data('overall','merged','SAXS',self.quadrant_unfold(merged).d)
        self.save_data(['overall'],['merged'])
        self.set_h5_attr(f"overall/merged",'merge_params', json.dumps(merge_settings))
        self.set_h5_attr(f"overall/merged",'Nimages', len(qxy_list))
        self.set_h5_attr(f"overall/merged",'removal_index', removal_index)
        self.set_h5_attr(f"overall/merged",'em2_average', np.mean(em2_list))

        for box in params.boxes:
            self.set_h5_attr(f"overall/merged/{box['label']}",'box_params', json.dumps(box))


    def proc_muscle_livestreaming(self,sn,params):
        """
        standard muscle processing pipeline for live streaming data using preset parameters.
        Intended for quick checking of data and not of final processed data.
        """

        self.process_qxy_fiber(sn=sn,subpixel = params.subpixel,mask=params.mask,align_threshold = params.align_threshold)
        qxy_list = self.proc_data[sn]['processed']['qxy_SAXS'].copy() #list of data2s
        N = len(qxy_list)

        dd2 = qxy_list[0].copy()
        lines_equator = [None]*N
        lines_meridionals = [None]*N
        for i,qxy in enumerate(qxy_list):
            _,lines_equator[i]    = self.grab_fiber_line_profile(data2d = qxy,direction = 'x',qx_min = 0,qx_max = 1e2,
                                                                                              qy_min = 0,qy_max = 0.005 * c)
            _,lines_meridionals[i] = self.grab_fiber_line_profile(data2d = qxy,direction = 'y',qx_min = 0,qx_max = 0.005 * c,
                                                                                               qy_min = 0,qy_max = 1e2)
        dd2.d = np.array(lines_equator)
        self.add_proc_data(sn, 'processed', 'equators',dd2)
        dd2 = dd2.copy() 
        dd2.d = np.array(lines_meridionals)
        self.add_proc_data(sn, 'processed', 'meridionals',dd2)
        self.save_data([sn], ['processed'])
        
        #Blind merging for quick checks
        merged = self.Merge_qxy(qxy_list)
        self.add_proc_data(sn, 'processed','merged_qxy_SAXS' , self.quadrant_unfold(merged))
        self.save_data([sn], ['processed'],['merged_qxy_SAXS'])
        self.set_h5_attr(f"{sn}/processed/{'merged_qxy_SAXS'}", "Nimages", len(qxy_list))

        return None

    @h5_file_access
    def process_qxy_fiber(self,sn,subpixel=True,mask = None,debug=True,align_threshold = 5):
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
                im,rotated_negativemask,phi = Align(im*negativemask,negativemask,align_threshold = align_threshold)
                
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


    def Merge_qxy(self,qxy_list):
        N = len(qxy_list)       

        merged = MatrixWithCoords()
        merged.xc_label = "qx"
        merged.xc = qxy_list[0].xc
        merged.yc_label = "qy"
        merged.yc = qxy_list[0].yc
        merged.d = np.zeros_like(qxy_list[0].d)
        merged = merged.merge(qxy_list)

        return merged

    def quadrant_unfold(self,qxy): 
        #qxy is the +/+ quadrant. This function unfolds it into four quadrants
        n,m = qxy.d.shape
        new = np.zeros((2*n-1,2*m-1))
        new[:n,:m] = np.flipud(np.fliplr(qxy.d))
        new[:n,m:] = np.flipud(qxy.d[:,:-1])
        new[n:,:m] = np.fliplr(qxy.d[:-1,:])
        new[n:,m:] = qxy.d[:-1,:-1]
        qxy.d = new
        return qxy

    def link_2_raw(self,fns):
        self.enable_write(True)
        for s in fns:
            with h5py.File(s, "r", swmr=True) as fs:
                for sn in fs.keys():
                    for dk in fs[sn].keys():
                        dkk = f"{sn}/{dk}"
                        print(dkk)
                        self.fh5[dkk] = h5py.ExternalLink(s, dkk)
        self.enable_write(False)

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



def monotonic(q,y):
    z = y
    q = np.append(q,[0.99*q.min(),1.01*q.max()])
    z = np.append(z,[2*z.max(),2*z.max()])
    points = np.array([q,z]).transpose()
    hull = ConvexHull(points)
    hullpoints = np.array([[points[vertex, 0], points[vertex, 1]] for vertex in hull.vertices ])
    h = interp1d(hullpoints[:,0],hullpoints[:,1],bounds_error=False)
    return h
