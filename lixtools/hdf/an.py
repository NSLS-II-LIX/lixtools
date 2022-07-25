import h5py,time
import numpy as np
import multiprocessing as mp
import json,os,copy
import fast_histogram as fh

from py4xs.data2d import Data2d,MatrixWithCoords,unflip_array
from py4xs.slnxs import Data1d
from py4xs.hdf import lsh5,h5xs,proc_merge1d
from py4xs.utils import get_bin_ranges_from_grid

save_fields = {"Data1d": {"shared": ['qgrid'], "unique": ["data", "err", "trans", "trans_e", "trans_w"]},
               "MatrixWithCoords": {"shared": ["xc", "yc", "xc_label", "yc_label"], "unique": ["d", "err"]},
               "ndarray": {"shared": [], "unique": []}
              }

def regularize(ar, prec):
    """ adjust array elements so that they are evenly spaced
        and limit digits to the specified precision
    """
    step = np.mean(np.diff(ar))
    step = prec*np.floor(np.fabs(step)/prec+0.5)*np.sign(step)
    off = np.mean(ar-np.arange(len(ar))*step)
    off = prec*10*np.floor(np.fabs(off)/prec/10+0.5)*np.sign(off)
    return off+np.arange(len(ar))*step
        
def get_scan_parms(dh5xs, sn, prec=0.0001, force_uniform_steps=True):
    """ figure out the scan shape and motor positions, assuming 2D grid scans 
        i.e. sufficient to have a single set of x and y coordinates to specify the location
    """
    shape = dh5xs.header(sn)['shape']
    assert(len(shape)==2)
    dec = -int(np.log10(prec))

    motors = dh5xs.header(sn)['motors']
    pn = dh5xs.header(sn)['plan_name']
    if pn=="raster":
        snaking = True
    elif "snaking" in dh5xs.header(sn).keys():
        snaking = dh5xs.header(sn)["snaking"][-1]
    else: 
        snaking = False

    if len(motors)!=2:
        raise Exception(f"expecting two motors, got {motors}.")
    # slow axis is the first motor 
    spos = dh5xs.fh5[sn][f"primary/data/{motors[0]}"][...].flatten() 
    n = int(len(spos)/shape[0])
    if n>1:
        spos = spos[::n]         # in step scans, the slow axis position is reported every step
    if force_uniform_steps:
        spos = regularize(spos, prec)
    spos = spos.round(dec)
    
    # for the fast axis, the Newport fly scan sometime repeats position data  
    fpos = dh5xs.fh5[sn][f"primary/data/{motors[1]}"][...].flatten()
    n = int(len(fpos)/shape[0]/shape[1])
    fpos = fpos[::n]         # remove redundancy, specific to fly scanning with Newport
    fpos = fpos[:shape[1]]   # assume these positions are repeating
    if force_uniform_steps:
        fpos = regularize(fpos, prec)
    fpos = fpos.round(dec)
        
    return {"shape": shape, "snaking": snaking, 
            "fast_axis": {"motor": motors[1], "pos": list(fpos)}, 
            "slow_axis": {"motor": motors[0], "pos": list(spos)}}  # json doesn't like numpy arrays



def proc_merge1d(args):
    """ utility function to perfrom azimuthal average and merge detectors
    """
    images,sn,nframes,starting_frame_no,debug,detectors,qgrid = args
    ret = []
    
    if debug is True:
        print("processing started: sample = %s, starting frame = #%d \r" % (sn, starting_frame_no))
    for i in range(nframes):
        d1s = []
        for det in detectors:
            dt = Data1d()
            label = "%s_f%05d%s" % (sn, i+starting_frame_no, det.extension)
            dt.load_from_2D(images[det.extension][i], det.exp_para, qgrid, 
                            pre_process=det.pre_process, flat_cor=det.flat, mask=det.exp_para.mask,
                            save_ave=False, label=label)
            if det.dark is not None:
                dt.data -= det.dark
            dt.scale(1./det.fix_scale)
            d1s.append(dt)
    
        dm = d1s[0]
        for dm1 in d1s[1:]:
            dm.merge(dm1)
        ret.append(dm)
            
    if debug is True:
        print(f"processing completed: {sn}, {starting_frame_no}.                          \r", end="")

    return [sn, starting_frame_no, ret]


def proc_merge2d(args):
    fn,img_grps,sn,debug,parms,bin_ranges = args
    
    fh5 = h5py.File(fn, "r", swmr=True)
    if debug is True:
        print(f"processing started: {sn}            \r", end="")

    s = fh5[img_grps[0]].shape
    nframes = len(np.ones(s[:-2]).flatten())
    frns = np.arange(nframes, dtype=int).reshape(s[:-2])
    dQ,dPhi,dMask,dQPhiMask,dWeight = parms
    ndet = len(dQ)
    
    cQPhiMask = ~dQPhiMask[0]
    for i in range(1,ndet):
        cQPhiMask &= ~dQPhiMask[i]   
    q_bin_ranges,q_N,phi_range,phi_N = bin_ranges       
    ret = np.zeros(shape=(nframes, phi_N, q_N))
    
    for i in range(ndet):
        data = fh5[img_grps[i]]
        for idx,n in np.ndenumerate(frns):
        # there should be no overlap between SAXS and WAXS, not physically possible
            if n%500==0 and debug:
                print(f"- {img_grps[i]}, {idx}            \r", end="")
            mm = np.vstack([fh.histogram2d(dQ[i], dPhi[i], bins=(qN, phi_N), range=[qrange, phi_range], 
                                           weights=data[idx][dMask[i]]) for qrange,qN in q_bin_ranges]).T
            mm[dQPhiMask[i]]/=dWeight[i][dQPhiMask[i]]
            ret[n][mm>0] = mm[mm>0]

    for n in range(nframes):
        ret[n][cQPhiMask] = np.nan
    if debug is True:
        print(f"processing completed: {sn}               \r", end="")
        
    fh5.close()
    return [sn, ret]

def proc_merge2d0(args):
    images,sn,starting_frame_cnt,debug,parms,bin_ranges = args
    fn,img_grps,idx,nstart,nframes = images   
    
    fh5 = h5py.File(fn, "r", swmr=True)
    if debug is True:
        print(f"processing started: {sn}, frame #{starting_frame_cnt}            \r", end="")

    dQ,dPhi,dMask,dQPhiMask,dWeight = parms
    ndet = len(dQ)

    cQPhiMask = ~dQPhiMask[0]
    for i in range(1,ndet):
        cQPhiMask &= ~dQPhiMask[i]   
    q_bin_ranges,q_N,phi_range,phi_N = bin_ranges       
    ret = np.zeros(shape=(nframes, phi_N, q_N))
    
    for i in range(ndet):
        data = fh5[img_grps[i]][idx][nstart:nstart+nframes]
        if debug:
            print(f"- {img_grps[i]}, {idx}, {nstart}:{nstart+nframes}      \r", end="")
        for n in range(nframes):
        # there should be no overlap between SAXS and WAXS, not physically possible
            mm = np.vstack([fh.histogram2d(dQ[i], dPhi[i], bins=(qN, phi_N), range=[qrange, phi_range], 
                                           weights=data[n][dMask[i]]) for qrange,qN in q_bin_ranges]).T
            mm[dQPhiMask[i]]/=dWeight[i][dQPhiMask[i]]
            ret[n][mm>0] = mm[mm>0]

    for n in range(nframes):
        ret[n][cQPhiMask] = np.nan
    if debug is True:
        print(f"processing completed: {sn}, frame #{starting_frame_cnt}               \r", end="")
        
    fh5.close()
    return [sn, starting_frame_cnt, ret]

def update_res_path(res_path, replace_res_path={}):
    for rp1,rp2 in replace_res_path.items():
        print("updating resource path ...")
        if rp1 in res_path:
            res_path = res_path.replace(rp1, rp2)  
    return res_path

class h5xs_an(h5xs):
    """ keep the detector information
        import data from raw h5 files, keep track of the file location
        copy the meta data, convert raw data into q-phi maps or azimuthal profiles
        can still show data, through the h5xs object for the raw data
        
        meta data such as scan or HPLC info should be dealt with separately
        saving and loading
        
    """
    def __init__(self, *args, 
                 Nphi=32, load_raw_data=True, pre_proc="2D", replace_path={}, 
                 **kwargs):
        """ pre_proc: should be either 1D or 2D, determines whether to save q-phi maps or 
                azimuhtal averages as the pre-processed data
            replace_path: should be a dictionary {old_path: new_path}
                this is useful when the source raw data files have been moved
        """        
        fn = args[0]  # any better way to get this?
        if not os.path.exists(fn):
            h5py.File(fn, 'w').close()
        super().__init__(*args, have_raw_data=False, **kwargs)
        self.proc_data = {}
        self.h5xs = {}
        self.raw_data = {}
        
        self.enable_write(True)
        if "pre_proc" in self.fh5.attrs:
            pre_proc0 = self.fh5.attrs["pre_proc"]
            if pre_proc0!=pre_proc:
                print(f"Warning: using existing pre_proc type {pre_proc0} instead of {pre_proc}")
                pre_proc = pre_proc0
        else:
            self.fh5.attrs["pre_proc"] = pre_proc    
        self.pre_proc = pre_proc
        if pre_proc=="2D":
            if "Nphi" in self.fh5.attrs:
                Nphi0 = int(self.fh5.attrs['Nphi'])
                if Nphi0!=Nphi:
                    print(f"Warning: using existing Nphi={Nphi0} instead of {Nphi}")
            else:
                self.fh5.attrs['Nphi'] = Nphi
            self.Nphi = Nphi
            self.phigrid = np.linspace(-180, 180, self.Nphi)
        self.enable_write(False)
        
        # if there are raw data file info, prepare the read only h5xs objects in case needed
        self.samples = list(self.fh5.keys())
        if 'overall' in self.samples:
            self.samples.remove('overall')
            self.attrs['overall'] = {}
        for sn in self.samples:
            self.attrs[sn] = {}
            fn_raw0 = self.fh5[sn].attrs['source']
            fn_raw_path = os.path.dirname(fn_raw0)
            fn_raw = update_res_path(fn_raw0, replace_path)
            if not os.path.exists(fn_raw):
                raise Exception(f"raw data file {fn_raw} does not exist ...")
            if fn_raw!=fn_raw0:
                self.enable_write(True)
                self.fh5[sn].attrs['source'] = fn_raw
                self.enable_write(False)
            if load_raw_data:
                if not fn_raw in self.raw_data.keys():
                    self.raw_data[fn_raw] = h5xs(fn_raw, [self.detectors, self.qgrid], read_only=True)
                self.h5xs[sn] = self.raw_data[fn_raw]
            else: 
                self.h5xs[sn] = fn_raw
            self.attrs[sn]['header'] = json.loads(self.fh5[sn].attrs['header'])

    def list_data(self):
        for sn,d in self.proc_data.items(): 
            print(sn)
            for dk,dd in d.items(): # data key
                print("++", dk)  # should also print data type
                for sk,sd in dd.items(): # sub key
                    print("++++", sk)  # should also print data size

    def show_data(self, sn, **kwargs):
        return self.h5xs[sn].show_data(sn=sn, detectors=self.detectors, **kwargs)

    def show_data_qphi(self, sn, **kwargs):
        return self.h5xs[sn].show_data_qphi(sn=sn, detectors=self.detectors, **kwargs)

    def show_data_qxy(self, sn, **kwargs):
        return self.h5xs[sn].show_data_qxy(sn=sn, detectors=self.detectors, **kwargs)

    def import_raw_data(self, fn_raw, sn=None, save_attr=["source", "header"], debug=False, **kwargs):
        """ create new group, copy header
            save_attr: meta data that should be extracted from the raw data file
                       for scanning data, this should be ["source", "header", 'scan']
                       for HPLC data ???
        """
        if debug:
            print(f"importing meta data from {fn_raw} ...")
        
        dt = h5xs(fn_raw, [self.detectors, self.qgrid], read_only=True)
        if sn is None:
            sns = dt.samples
        else: 
            if not sn in dt.samples:
                raise Exception(f"cannot find data on {sn} in {fn_raw}.")
            sns = [sn]
        
        for sn in sns:
            self.h5xs[sn] = dt
            self.enable_write(True)
            if not sn in self.attrs.keys():
                self.attrs[sn] = {}
            if not sn in self.fh5.keys():
                grp = self.fh5.create_group(sn)
            else:
                grp = self.fh5[sn]

            if "source" in save_attr:
                self.attrs[sn]['source'] = os.path.realpath(fn_raw)
                grp.attrs['source'] = self.attrs[sn]['source'] 
            if "header" in save_attr:
                self.attrs[sn]['header'] = dt.header(sn)
                grp.attrs['header'] = json.dumps(self.attrs[sn]['header'])
            if "scan" in save_attr:
                self.attrs[sn]['scan'] = get_scan_parms(dt, sn, **kwargs)
                grp.attrs['scan'] = json.dumps(self.attrs[sn]['scan'])

            self.enable_write(False)
        
            if not sn in self.proc_data.keys():
                self.proc_data[sn] = {}
            self.list_samples(quiet=True)
        
        return sns

    def get_mon(self, sn, *args, **kwargs):
        if not sn in self.h5xs.keys():
            raise Exception(f"no raw data on {sn}.")
        self.h5xs[sn].get_mon(sn, *args, **kwargs)
        if 'attrs' not in self.proc_data[sn].keys():
            self.proc_data[sn]['attrs'] = {}
        self.proc_data[sn]['attrs']["transmitted"] = self.h5xs[sn].d0s[sn]["transmitted"]
        self.proc_data[sn]['attrs']["incident"] = self.h5xs[sn].d0s[sn]["incident"]
        self.proc_data[sn]['attrs']["transmission"] = self.h5xs[sn].d0s[sn]["transmission"]
    
    def has_proc_data(self, sn, data_key, sub_key):
        if not sn in self.proc_data.keys():
            return False
        if not data_key in self.proc_data[sn].keys():
            return False
        if not sub_key in self.proc_data[sn][data_key].keys():
            return False
        return True
        
    def add_proc_data(self, sn, data_key, sub_key, data):
        if sn not in self.proc_data.keys():
            self.proc_data[sn] = {}
        if data_key not in self.proc_data[sn].keys():
            self.proc_data[sn][data_key] = {}
        self.proc_data[sn][data_key][sub_key] = data
    
    def extract_attr(self, sn, attr_name, func, data_key, sub_key, N=8, **kwargs):
        """ extract an attribute from the pre-processed data using the specified function
            and source of the data (data_key/sub_key)
        """
        data = [func(d, **kwargs) for d in self.proc_data[sn][data_key][sub_key]]
        self.add_proc_data(sn, 'attrs', attr_name, np.array(data))
        
    def process(self, N=8, max_c_size=1024, debug=True):
        if debug is True:
            t1 = time.time()
            print("processing started, this may take a while ...")                

        if self.pre_proc=="1D":
            self.process1d(N, max_c_size, debug)
        elif self.pre_proc=="2D":
            if len(self.h5xs)>N/2: # one process per sample
                self.process2d(N, max_c_size, debug)
            else: # single sample, split data 
                self.process2d0(N, max_c_size, debug)
        else:
            raise Exception(f"cannot deal with pre_proc = {self.pre_proc}")

        if debug is True:
            t2 = time.time()
            print("done, time elapsed: %.2f sec" % (t2-t1))                

    def process1d(self, N=8, max_c_size=1024, debug=True):
        qgrid = self.qgrid 
        detectors = self.detectors              
                
        results = {}
        pool = mp.Pool(N)
        jobs = []
        
        for sn in self.h5xs.keys():
            results[sn] = {}
            if debug:
                print(f"processing {sn} ...")
            dh5 = self.h5xs[sn]

            s = dh5.dset(dh5.det_name[self.detectors[0].extension]).shape
            if len(s)==3 or len(s)==4:
                n_total_frames = s[-3]  # fast axis
            else:
                raise Exception("don't know how to handle shape:", )
            if n_total_frames<N*N/2:
                Np = 1
                c_size = N
            else:
                Np = N
                c_size = int(n_total_frames/N)
                if max_c_size>0 and c_size>max_c_size:
                    Np = int(n_total_frames/max_c_size)+1
                    c_size = int(n_total_frames/Np)
            
            if debug:
                print(f"sn={sn}, Np={Np}, c_size={c_size}")
            
            t = np.ones(s[:-3])
            # process data in group in hope to limit memory use
            # the raw data could be stored in a 1d or 2d array
            fcnt = 0
            for i in range(Np):
                if i==Np-1:
                    nframes = n_total_frames - c_size*(Np-1)
                else:
                    nframes = c_size
                
                for idx, x in np.ndenumerate(t):  # idx should enumerate the outter-most indices
                    images = {det.extension:
                              dh5.dset(dh5.det_name[det.extension])[idx][i*c_size:i*c_size+nframes] for det in detectors}
                    if N>1: # multi-processing, need to keep track of total number of active processes
                        job = pool.map_async(proc_merge1d, [(images, sn, nframes, fcnt, 
                                                             debug, detectors, self.qgrid)])
                        jobs.append(job)
                    else: # serial processing
                        [sn, fr1, data] = proc_merge1d((images, sn, nframes, fcnt,
                                                        debug, detectors, self.qgrid))
                        results[sn][fr1] = data
                    fcnt += nframes
                    
        if N>1:             
            for job in jobs:
                [sn, fr1, data] = job.get()[0]
                results[sn][fr1] = data
                print("data received: sn=%s, fr1=%d" % (sn,fr1) )
            pool.close()
            pool.join()

        for sn in self.samples:
            if sn not in results.keys():
                continue
            data = []
            frns = list(results[sn].keys())
            frns.sort()
            for frn in frns:
                data.extend(results[sn][frn])
            self.add_proc_data(sn, 'azi_avg', 'merged', data)     

    def process2d(self, N=8, max_c_size=1024, debug=True):
        """ produce merged q-phi maps
            the bottleneck is reading the data
        """
        qgrid = self.qgrid 
        phigrid = self.phigrid
        detectors = self.detectors
        phi_range = [-180, 180]
        phi_N = self.Nphi
        q_bin_ranges = get_bin_ranges_from_grid(qgrid)
        bin_ranges = [q_bin_ranges,len(qgrid),phi_range,phi_N] 
        
        print("this might take a while ...")
        
        # prepare the info needed for processing
        dQ = {}
        dPhi = {}
        dMask = {}
        dQPhiMask = {}
        dWeight = {}
        
        for i in range(len(detectors)):
            exp = detectors[i].exp_para
            dMask[i] = ~unflip_array(exp.mask.map, exp.flip)
            dQ[i] = unflip_array(exp.Q, exp.flip)[dMask[i]]
            dPhi[i] = unflip_array(exp.Phi, exp.flip)[dMask[i]]
            ones = np.ones_like(dQ[i])
            dWeight[i] = np.vstack([fh.histogram2d(dQ[i], dPhi[i], bins=(qN, phi_N), range=[qrange, phi_range], weights=ones) 
                                 for qrange,qN in q_bin_ranges]).T
            dWeight[i] *= detectors[i].fix_scale
            dQPhiMask[i] = (dWeight[i]>0)
        parms = [dQ,dPhi,dMask,dQPhiMask,dWeight]
        
        # for corrections: polarization and solid angle 
        #QPhiCorF = np.ones_like()

        N = np.min([N, len(self.h5xs)])        
        pool = mp.Pool(N)
        jobs = []
        results = {}   # scanning data sets are too large, process one sample at a time
        for sn,dh5 in self.h5xs.items():
            img_grps = [dh5.dset(dh5.det_name[det.extension], get_path=True, sn=sn) for det in detectors]
            
            job = pool.map_async(proc_merge2d, [(dh5.fn, img_grps, sn, debug, parms, bin_ranges)])  
            jobs.append(job)
                
        pool.close()
        for job in jobs:
            [sn, data] = job.get()[0]
            results[sn] = data
            print(f"data received: sn={sn}                \r", end="")
        pool.join()

        for sn in results.keys():
            data = []
            for i in range(len(results[sn])):
                dd2 = MatrixWithCoords()
                dd2.xc = qgrid
                dd2.xc_label = "q"
                dd2.yc = phigrid
                dd2.yc_label = "phi"
                dd2.d = results[sn][i]
                dd2.err = None
                data.append(dd2)
            self.add_proc_data(sn, 'qphi', 'merged', data)            
            
            
    def process2d0(self, N=8, max_c_size=1024, debug=True):
        """ produce merged q-phi maps
            for large files, the bottleneck is reading the data
            mult
        """
        qgrid = self.qgrid 
        phigrid = self.phigrid
        detectors = self.detectors
        phi_range = [-180, 180]
        phi_N = self.Nphi
        q_bin_ranges = get_bin_ranges_from_grid(qgrid)
        bin_ranges = [q_bin_ranges,len(qgrid),phi_range,phi_N] 
        
        print("this might take a while ...")
        
        # prepare the info needed for processing
        dQ = {}
        dPhi = {}
        dMask = {}
        dQPhiMask = {}
        dWeight = {}
        
        for i in range(len(detectors)):
            exp = detectors[i].exp_para
            dMask[i] = ~unflip_array(exp.mask.map, exp.flip)
            dQ[i] = unflip_array(exp.Q, exp.flip)[dMask[i]]
            dPhi[i] = unflip_array(exp.Phi, exp.flip)[dMask[i]]
            ones = np.ones_like(dQ[i])
            dWeight[i] = np.vstack([fh.histogram2d(dQ[i], dPhi[i], bins=(qN, phi_N), range=[qrange, phi_range], weights=ones) 
                                 for qrange,qN in q_bin_ranges]).T
            dWeight[i] *= detectors[i].fix_scale
            dQPhiMask[i] = (dWeight[i]>0)
        parms = [dQ,dPhi,dMask,dQPhiMask,dWeight]
        
        # for corrections: polarization and solid angle 
        #QPhiCorF = np.ones_like()

                
        for sn in self.h5xs.keys():
            if N>1:
                pool = mp.Pool(N)
                jobs = []
            results = {}   # scanning data sets are too large, process one sample at a time
            if debug:
                print(f"processing {sn} ...")
            dh5 = self.h5xs[sn]
            s = dh5.dset(dh5.det_name[self.detectors[0].extension]).shape
            img_grp = [dh5.dset(dh5.det_name[det.extension], get_path=True, sn=sn) for det in detectors]
            #dh5.fh5.close()
            
            if len(s)==3 or len(s)==4:
                n_total_frames = s[-3]  # fast axis
            else:
                raise Exception("don't know how to handle shape:", s)
            if n_total_frames<N*N/2:
                Np = 1
                c_size = N
            elif len(s)==4:
                if n_total_frames<max_c_size or max_c_size<=0:
                    c_size = n_total_frames
                    Np = 1
                else:
                    Np = int(n_total_frames/max_c_size)+1
                    c_size = int(n_total_frames/Np)
            else:
                Np = N
                c_size = int(n_total_frames/N)
                if max_c_size>0 and c_size>max_c_size:
                    Np = int(n_total_frames/max_c_size)+1
                    c_size = int(n_total_frames/Np)
                        
            t = np.ones(s[:-3])
            fcnt = 0
            for i in range(Np):
                if i==Np-1:
                    nframes = n_total_frames - c_size*(Np-1)
                else:
                    nframes = c_size
                
                for idx, x in np.ndenumerate(t):  # idx should enumerate the outter-most indices
                    images = [dh5.fn, img_grp, idx, i*c_size, nframes]
                    if N>1: # multi-processing, need to keep track of total number of active processes
                        job = pool.map_async(proc_merge2d0, [(images, sn, fcnt, debug, parms, bin_ranges)])  
                        jobs.append(job)
                    else: # serial processing
                        [sn, fr1, data] = proc_merge2d0((images, sn, fcnt, debug, parms, bin_ranges))
                        results[fr1] = data
                    fcnt += nframes
                
            if N>1: 
                for job in jobs:
                    [sn, fr1, data] = job.get()[0]
                    results[fr1] = data
                    print(f"data received: sn={sn}, fr1={fr1}                \r", end="")
                pool.close()
                pool.join()

            data = []
            for k in sorted(results.keys()):
                for i in range(len(results[k])):
                    dd2 = MatrixWithCoords()
                    dd2.xc = qgrid
                    dd2.xc_label = "q"
                    dd2.yc = phigrid
                    dd2.yc_label = "phi"
                    dd2.d = results[k][i]
                    dd2.err = None
                    data.append(dd2)
            self.add_proc_data(sn, 'qphi', 'merged', data)

    def export_data(self, fn):
        """ export all data under "overall" if there are multiple samples
        """
        if len(self.h5xs)==1:
            sn = self.samples[0]
        elif 'overall' not in self.proc_data.keys():
            raise Exception(f"Expecting overall data but not found ...")
        else:
            sn = "overall"
        
        fh5 = h5py.File(fn, "w-")  # fail if the file exists
        dks = self.proc_data[sn].keys()
        for data_key in dks:
            sks = self.proc_data[sn][data_key].keys()
            for sub_key in sks:
                print(f"{sn}, {data_key}, {sub_key}        \r", end="")
                self.pack(sn, data_key, sub_key, fh5=fh5)
        fh5.close()
    
    def save_data(self, save_sns=None, save_data_keys=None, save_sub_keys=None):
        print("saving processed data ...")
        if save_sns is None:
            save_sns = list(self.fh5.keys())
            if "overall" in self.proc_data.keys():
                save_sns += ["overall"]
        elif not isinstance(save_sns, list):
            save_sns = [save_sns]
            
        self.enable_write(True)
        for sn in save_sns:
            dks = self.proc_data[sn].keys()
            if save_data_keys is not None:
                if not isinstance(save_data_keys, list):
                    save_data_keys = [save_data_keys]
                dks = list(set(save_data_keys) & set(dks))
            for data_key in dks:
                sks = self.proc_data[sn][data_key].keys()
                if save_sub_keys is not None:
                    if not isinstance(save_sub_keys, list):
                        save_sub_keys = [save_sub_keys]
                    sks = list(set(save_sub_keys) & set(sks))
                for sub_key in sks:
                    print(f"{sn}, {data_key}, {sub_key}        \r", end="")
                    self.pack(sn, data_key, sub_key)
        self.fh5.flush()
        self.enable_write(False)
        print("done.                      ")
    
    def pack(self, sn, data_key, sub_key, fh5=None):
        """ this is for packing processed data, which should be stored under self.proc_data as a dictionary
            sn="overall" is merged from all samples in the h5xs_an object
            the key is the name/identifier of the processed data
                e.g. attrs, qphi, azi_avg, maps, tomo 
            all the data stored under the data_key are of the same data type 
            "attrs" are arrays, for values derived from the raw data file (e.g beam intensity), save as is
            all other data should be either Data1d or MatrixWithCoords, with the same 
                "shared" properties, saved as attributes of the datagroup
            the sub_key may not always be necessary, but is required for consistency 
                e.g. "transmission", "merged", "subtracted", "bkg" 

            pack_data() saves the data into the h5 file under the group processed/{data_key}/{sub_key}
        """
        data = self.proc_data[sn][data_key][sub_key]
        if isinstance(data, list):
            d0 = data[0]
            n = len(data)
        else:
            d0 = data
            n = 1

        dtype = d0.__class__.__name__
        if not dtype in save_fields.keys():
            print(sn,data_key,sub_key)
            raise Exception(f"{dtype} is not supported for packing.")

        if fh5 is None:
            fh5 = self.fh5
        # the group fh5[sn] should already exist, created when importing raw data
        # except for "overall"
        if not sn in fh5.keys():
            grp = fh5.create_group(sn)
        else:
            grp = fh5[sn]

        # if the data group exists, the saved attributes needs to match those for the new data, 
        # otherwise raise exception, in case there is a conflict with existing data
        # the "shared" fields of the data, e.g. qgrid in azimuthal average, are then saved as attributes of the data group
        if data_key in list(grp.keys()):
            grp = grp[data_key]
            if grp.attrs['type']!=dtype:
                raise Exception(f"data type of {data_key} for {sn} does not match existing data")
            for k in save_fields[dtype]["shared"]:
                if isinstance(d0.__dict__[k], str):  # labels
                    if d0.__dict__[k]==grp.attrs[k]:
                        continue
                elif np.equal(d0.__dict__[k], grp.attrs[k]).all():
                    continue
                raise Exception(f"{k} in {data_key} for {sn} does not match existing data")
        else:
            grp = grp.create_group(data_key)
            grp.attrs['type'] = dtype
            for k in save_fields[dtype]["shared"]:
                grp.attrs[k] = d0.__dict__[k]

        # under the group, save the "unique" fields as datasets, named as sub_key.unique_field
        # write numpy array as is
        if dtype=="ndarray":
            sd = self.proc_data[sn][data_key][sub_key]
            if sd is not None:  # e.g. err for some MatrixWithCoords
                if sub_key in grp.keys():
                    grp[sub_key][...] = sd
                else:
                    grp.create_dataset(sub_key, data=sd)
                return

        if not sub_key in list(grp.keys()):
            grp.create_group(sub_key)
        grp = grp[sub_key]
        grp.attrs['len'] = n
        data = self.proc_data[sn][data_key][sub_key]
        for k in save_fields[dtype]['unique']:
            if not k in d0.__dict__.keys():
                continue
            if d0.__dict__[k] is None:  # e.g. err in MatrixWithCoords
                continue
            if n==1:
                sd = d0.__dict__[k]
            else:
                sd = np.array([d.__dict__[k] for d in data])
            if k in grp.keys():
                grp[k][...] = sd
            else:
                grp.create_dataset(k, data=sd)
    
    def load_data(self, samples=None):
        if samples is None:
            samples = self.fh5.keys()
        elif isinstance(samples, str):
            samples = [samples]
        for sn in samples:
            if not sn in self.proc_data.keys():
                print(f"loading data for {sn}")
                self.proc_data[sn] = {}
            for data_key in self.fh5[sn].keys():
                dtype = self.fh5[sn][data_key].attrs['type']
                if not data_key in self.proc_data[sn].keys():
                    self.proc_data[sn][data_key] = {}
                if dtype=='Data1d':
                    d0 = Data1d()
                elif dtype=="MatrixWithCoords":
                    d0 = MatrixWithCoords()
                for field in save_fields[dtype]['shared']:
                    d0.__dict__[field] = self.fh5[sn][data_key].attrs[field]
                for sub_key in self.fh5[sn][data_key].keys():
                    print(f"{data_key} ({dtype}): {sub_key}                \r", end="")
                    h5data = self.fh5[sn][data_key][sub_key]
                    if dtype=="ndarray":
                        self.proc_data[sn][data_key][sub_key] = h5data[...]
                    else:
                        fields = save_fields[dtype]['unique']
                        n = h5data.attrs['len']
                        if n==1:
                            data = copy.copy(d0)
                            for f in fields:
                                if not f in h5data.keys():
                                    continue
                                data.__dict__[f] = h5data[f][...]
                        else:
                            data = [copy.copy(d0) for _ in range(n)]
                            for f in fields:
                                if not f in h5data.keys():
                                    continue 
                                for i in range(n):
                                    data[i].__dict__[f] = h5data[f][i]
                        self.proc_data[sn][data_key][sub_key] = data
        print("done.                                           ")
    
    def qphi_bkgsub(self, bsn, bfrns):
        """ background subtraction, using the specified sample name and frame numbers 
            also correct/normalize for transmitted intensity, using the bkg trans as the reference value
            this value is saved as ['attrs']['ref_trans']
        """
        if bsn not in self.samples:
            raise Exception(f"sample {bsn} does not exist.")
        if isinstance(bfrns, int):
            bfrns = [bfrns]
        n = 0
        dbkg = self.proc_data[bsn]['qphi']['merged'][bfrns[0]]
        if len(bfrns)>1:
            dbkg.d = np.nanmean([self.proc_data[bsn]['qphi']['merged'][i].d for i in bfrns], axis=0)
            dbkg.err = np.nanmean([self.proc_data[bsn]['qphi']['merged'][i].err for i in bfrns], axis=0)
            dbkg.err /= np.sqrt(len(bfrns))

        # there could be intensity in the sample data but not in the bkg
        dbkg.d[np.isnan(dbkg.d)] = 0
        dbkg.err[np.isnan(dbkg.err)] = 0
        mon_t = np.average(self.proc_data[bsn]['attrs']['transmitted'][bfrns])
        mon_i = np.average(self.proc_data[bsn]['attrs']['incident'][bfrns])

        for sn in self.samples:
            self.proc_data[sn]['attrs']['ref_trans'] = np.array([mon_t])
            self.proc_data[sn]['qphi']['bkg'] = dbkg
            self.proc_data[sn]['qphi']['subtracted'] = []
            for trans,mm in zip(self.proc_data[sn]['attrs']['transmitted'], self.proc_data[sn]['qphi']['merged']):
                m1 = mm.copy()
                sc = mon_t/trans
                m1.d = mm.d*sc - dbkg.d
                m1.err = mm.err*sc   # needs work here
                self.proc_data[sn]['qphi']['subtracted'].append(m1)

    def prepare_attrs(self, attr_dict):
        """ attr_dict should be a dictionary that defines the attribute names and method of calculation
            e.g. {"int_cellulose": {func="get_roi", kwargs=[1.5, 1.6, 120, 180]}, 
                  "int_amorphous": [1.1, 1.3, -120, -60],
                  "int_saxs": [0.05, 0.2, -180, 180]}
        """