import h5py,time
import numpy as np
import multiprocessing as mp
import json,os,copy
import fast_histogram as fh

from py4xs.data2d import Data2d,MatrixWithCoords,unflip_array
from py4xs.slnxs import Data1d
from py4xs.hdf import lsh5,h5xs,proc_merge1d,h5_file_access
from py4xs.utils import get_bin_ranges_from_grid,calc_avg

save_fields = {"Data1d": {"shared": ['qgrid'], "unique": ["data", "err"]},
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
    #spos = dh5xs.fh5[sn][f"primary/data/{motors[0]}"][...].flatten() 
    spos = dh5xs.dset(motors[0], sn=sn, return_reference=False).flatten() 
    n = int(len(spos)/shape[0])
    if n>1:
        spos = spos[::n]         # in step scans, the slow axis position is reported every step
    if force_uniform_steps:
        spos = regularize(spos, prec)
    spos = spos.round(dec)
    
    # for the fast axis, the Newport fly scan sometime repeats position data  
    #fpos = dh5xs.fh5[sn][f"primary/data/{motors[1]}"][...].flatten()
    fpos = dh5xs.dset(motors[1], sn=sn, return_reference=False).flatten()
    n = int(len(fpos)/shape[0]/shape[1])
    fpos = fpos[::n]         # remove redundancy, specific to fly scanning with Newport
    fpos = fpos[:shape[1]]   # assume these positions are repeating
    if force_uniform_steps:
        fpos = regularize(fpos, prec)
    fpos = fpos.round(dec)
        
    return {"shape": shape, "snaking": snaking, 
            "fast_axis": {"motor": motors[1], "pos": list(fpos)}, 
            "slow_axis": {"motor": motors[0], "pos": list(spos)}}  # json doesn't like numpy arrays


def proc_merge1d0(args):
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

def proc_merge1d(args):
    fn,img_grps,sn,frns,debug,parms,bin_ranges = args
    nframes = len(frns)

    fh5 = h5py.File(fn, "r", swmr=True)
    if debug is True:
        print(f"processing started: {sn}            \r", end="")
    dQ,dMask,cQMask,dWeight,scale = parms
    ndet = len(dQ)

    cQMask = ~(dWeight[0]>0)
    for i in range(1,ndet):
        cQMask &= ~(dWeight[i]>0)   
    q_bin_ranges,q_N = bin_ranges       

    ret = {}
    frn0 = frns[0]
    for i in range(ndet):
        data = fh5[img_grps[i]]
        ret[f'{i}d'] = np.zeros(shape=(nframes, q_N))
        ret[f'{i}e'] = np.zeros(shape=(nframes, q_N))
        for n in range(nframes):
            idx = n+frn0
            if n%500==0 and debug:
                print(f"- {img_grps[i]}, {idx}            \r", end="")
            mm = np.hstack([fh.histogram1d(dQ[i], bins=qN, range=qrange, 
                                           weights=data[idx][dMask[i]]) for qrange,qN in q_bin_ranges])
            mm2 = np.hstack([fh.histogram1d(dQ[i], bins=qN, range=qrange, 
                                            weights=data[idx][dMask[i]]**2) for qrange,qN in q_bin_ranges])
            ret[f'{i}d'][n] = mm/dWeight[i]
            ret[f'{i}e'][n] = np.sqrt(mm2-mm*mm)/dWeight[i]
    fh5.close()

    ret_d = np.zeros(shape=(nframes, q_N))
    ret_e = np.zeros(shape=(nframes, q_N))
    for n in range(nframes):
        dd = [ret[f'{i}d'][n]/scale[i] for i in range(ndet)]
        ee = [ret[f'{i}d'][n]/scale[i] for i in range(ndet)]
        ret_d[n],ret_e[n] = calc_avg(dd, ee, method="err_weighted")
        ret_d[n][cQMask] = np.nan
        ret_e[n][cQMask] = np.nan

    if debug is True:
        print(f"processing completed: {sn}               \r", end="")
    return [sn, frn0, ret_d, ret_e]

def proc_merge2d(args):
    #fn,img_grps,sn,debug,parms,bin_ranges = args
    fn,img_grps,sn,frns,debug,parms,bin_ranges = args
    dQ,dPhi,dMask,dQPhiMask,dWeight = parms
    nframes = len(frns)
    ndet = len(dQ)
    
    #if debug:
    #    print(img_grps,sn,frns,nframes)
    fh5 = h5py.File(fn, "r", swmr=True)
    if debug is True:
        print(f"processing started: {sn}            \r", end="")
    #s = fh5[img_grps[0]].shape
    
    cQPhiMask = ~dQPhiMask[0]
    for i in range(1,ndet):
        cQPhiMask &= ~dQPhiMask[i]   
    q_bin_ranges,q_N,phi_range,phi_N = bin_ranges       
    ret = np.zeros(shape=(nframes, phi_N, q_N))
    
    frn0 = frns[0]
    for i in range(ndet):
        data = fh5[img_grps[i]]
        for n in range(nframes):
            # there should be no overlap between SAXS and WAXS, not physically possible
            idx = n+frn0
            if n%500==0 and debug:
                print(f"- {img_grps[i]}, {idx}            \r", end="")
            mm = np.vstack([fh.histogram2d(dQ[i], dPhi[i], bins=(qN, phi_N), range=[qrange, phi_range], 
                                           weights=data[n+frn0][dMask[i]]) for qrange,qN in q_bin_ranges]).T
            mm[dQPhiMask[i]]/=dWeight[i][dQPhiMask[i]]
            ret[n][mm>0] = mm[mm>0]

    for n in range(nframes):
        ret[n][cQPhiMask] = np.nan
    if debug is True:
        print(f"processing completed: {sn}               \r", end="")
        
    fh5.close()
    return [sn, frn0, ret]

def proc_merge_qxy(args):
    fn,img_grps,sn,frns,debug,parms,q_N,qrange = args
    dQx,dQy,dMask,dQxyMask,dWeight = parms
    nframes = len(frns)
    ndet = len(dQx)
    
    #if debug:
    #    print(img_grps,sn,frns,nframes)
    fh5 = h5py.File(fn, "r", swmr=True)
    if debug is True:
        print(f"processing started: {sn}            \r", end="")
    #s = fh5[img_grps[0]].shape
    
    cQxyMask = ~dQxyMask[0]
    for i in range(1,ndet):
        cQxyMask &= ~dQxyMask[i]   
    ret = np.zeros(shape=(nframes, q_N, q_N))
    
    frn0 = frns[0]
    for i in range(ndet):
        data = fh5[img_grps[i]]
        for n in range(nframes):
            # there should be no overlap between SAXS and WAXS, not physically possible
            idx = n+frn0
            if n%500==0 and debug:
                print(f"- {img_grps[i]}, {idx}            \r", end="")
            mm = fh.histogram2d(dQx[i], dQy[i], bins=(q_N, q_N), 
                                range=(qrange, qrange), weights=data[n+frn0][dMask[i]]) 
            mm[dQxyMask[i]]/=dWeight[i][dQxyMask[i]]
            ret[n][mm>0] = mm[mm>0]

    for n in range(nframes):
        ret[n][cQxyMask] = np.nan
    if debug is True:
        print(f"processing completed: {sn}               \r", end="")
        
    fh5.close()
    return [sn, frn0, ret]

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

def update_res_path(res_path, replace_res_path={}, quiet=False):
    for rp1,rp2 in replace_res_path.items():
        if not quiet: 
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
                 Nphi=32, load_raw_data=True, ignore_source_path=False,
                 pre_proc="2D", replace_path={}, quiet=True,
                 **kwargs):
        """ pre_proc: should be either 1D or 2D, determines whether to save q-phi maps or 
                azimuhtal averages as the pre-processed data
            replace_path: should be a dictionary {old_path: new_path}
                this is useful when the source raw data files have been moved
        """        
        fn = args[0]  # any better way to get this?
        if not os.path.exists(fn):
            h5py.File(fn, 'w').close()
        super().__init__(*args, have_raw_data=False, exclude_sample_names=['overall'], **kwargs)
        self.proc_data = {}
        self.h5xs = {}
        self.raw_data = {}
        
        with h5py.File(fn) as self.fh5:            
            if "pre_proc" in self.fh5.attrs:
                pre_proc0 = self.fh5.attrs["pre_proc"]
                if pre_proc0!=pre_proc:
                    print(f"Warning: using existing pre_proc type {pre_proc0} instead of {pre_proc}")
                    pre_proc = pre_proc0
            else:
                self.enable_write(True)
                self.fh5.attrs["pre_proc"] = pre_proc
                self.enable_write(False)
            self.pre_proc = pre_proc
            if pre_proc=="2D":
                if "Nphi" in self.fh5.attrs:
                    Nphi0 = int(self.fh5.attrs['Nphi'])
                    if Nphi0!=Nphi:
                        print(f"Warning: using existing Nphi={Nphi0} instead of {Nphi}")
                        Nphi = Nphi0
                else:
                    self.enable_write(True)
                    self.fh5.attrs['Nphi'] =  Nphi
                    self.enable_write(False)
                self.Nphi = Nphi
                self.phigrid = np.linspace(-180, 180, self.Nphi)
        
            # if there are raw data file info, prepare the read only h5xs objects in case needed
            self.samples = list(self.fh5.keys())
            if 'overall' in self.samples:
                self.samples.remove('overall')
                self.attrs['overall'] = {}
            for sn in self.samples:
                self.attrs[sn] = {}
                fn_raw0 = self.fh5[sn].attrs['source']
                fn_raw_path = os.path.dirname(fn_raw0)
                fn_raw = update_res_path(fn_raw0, replace_path, quiet=quiet)
                if not os.path.exists(fn_raw):
                    if ignore_source_path:
                        fn_raw = ""
                    else:
                        raise Exception(f"raw data file {fn_raw} does not exist ...")
                if fn_raw!=fn_raw0:
                    self.enable_write(True)
                    self.fh5[sn].attrs['source'] = fn_raw
                    self.enable_write(False)
                if load_raw_data:
                    if not fn_raw in self.raw_data.keys():
                        self.raw_data[fn_raw] = h5xs(fn_raw, [self.detectors, self.qgrid], 
                                                     sn=sn, read_only=True)
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

    @h5_file_access
    def import_raw_data(self, fn_raw, sn=None, save_attr=["source", "header"], debug=False, **kwargs):
        """ create new group, copy header
            save_attr: meta data that should be extracted from the raw data file
                       for scanning data, this should be ["source", "header", 'scan']
                       for HPLC data ???
        """
        if not os.path.isfile(fn_raw):
            raise Exception(f"{fn_raw} is not a valid file name ...")
        if debug:
            print(f"importing meta data from {fn_raw} ...")
        
        dt = h5xs(fn_raw, [self.detectors, self.qgrid], sn=sn, read_only=True)
        if sn is None:
            sns = dt.samples
        else: 
            if not sn in dt.samples:
                raise Exception(f"cannot find data on {sn} in {fn_raw}.")
            sns = [sn]
        
        for sn in sns:
            if debug:
                print(f"    importing {sn} ...")
            self.h5xs[sn] = dt
            self.enable_write(True)
            if not sn in self.attrs.keys():
                self.attrs[sn] = {}
            if not sn in self.fh5.keys():
                grp = self.fh5.create_group(sn)
            else:
                grp = self.fh5[sn]

            if "source" in save_attr:
                # this seems out-of-place
                #if self.attrs[sn]['source']=="": # probably should have not gone this far anyway
                #    raise Exception(f"raw data for {sn} was never loaded ...")
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
    
    @h5_file_access
    def link_proc_data(self, fns):
        """ fns are expected to contain sample groups with processed data
            iterate over all samples, link every processed data groups under this an/an2.h5 file

            example of source dataset: {sn}/processed/qphi/merged/d 
            example of target dataset: {sn}/qphi/merged/d
        """
        #with h5py.File(self.fn, "w") as ff:
        self.enable_write(True)
        for s in fns:
            with h5py.File(s, "r", swmr=True) as fs:
                for sn in fs.keys():
                    if not sn in self.samples:
                        raise Exception(f"mismatched sample: {sn}")
                    if not "processed" in fs[sn].keys():
                        continue
                    for dk in fs[sn]["processed"].keys():
                        dkk0 = f"{sn}/processed/{dk}"
                        dkk = f"{sn}/{dk}"
                        print(dkk)
                        self.fh5[dkk] = h5py.ExternalLink(s, dkk0)
    
    @h5_file_access
    def extract_attr(self, sn, attr_name, func, data_key, sub_key, from_raw_data=False,
                     N=8, check_size=0, debug=False, **kwargs):
        """ extract an attribute from the pre-processed data using the specified function
            and source of the data (data_key/sub_key)
            check_size was a bandage solution and should be removed
        """
        if from_raw_data:
            rd = self.h5xs[sn].dset(sub_key, sn=sn, return_reference=False)
            data = np.array([func(d, **kwargs) for d in rd])
        else:
            data = np.array([func(d, **kwargs) for d in self.proc_data[sn][data_key][sub_key]])
        if debug:
            print(f"got data of shape {data.shape}")
        #if check_size>0:
        #    data = np.pad(data, (0,check_size-len(data)), constant_values=np.nan)
        if isinstance(attr_name, str):
            self.add_proc_data(sn, 'attrs', attr_name, np.array(data))
        elif isinstance(attr_name, list):
            if len(attr_name)!=data.shape[-1]:
                raise Exception(f"mismatch size of attr_name ({len(attr_name)}) and data {data.shape}")
            for i in range(len(attr_name)):
                #self.add_proc_data(sn, 'attrs', attr_name[i], data.reshape((-1, len(attr_name)))[:,i])  # this looks weird
                self.add_proc_data(sn, 'attrs', attr_name[i], data[..., i])   # preserve the shape of data
        else:
            raise Exception(f"don't know how to handle attr_name={attr_name}", quiet=True)
            
        self.save_data(save_sns=[sn], save_data_keys=['attrs'], save_sub_keys=attr_name)
        
    def process(self, N=8, max_c_size=1024, apply_symmetry=False, debug=True):
        """ generate the pre-processed data for downstream processing
            the type of data generated depends on self.pre_proc, either "1D" or "2D"
            with live data processing, this data may be available at the end of the scan
        """
        if debug is True:
            t1 = time.time()
            print("processing started, this may take a while ...")                

        if self.pre_proc=="1D":
            self.process1d(N, max_c_size, debug)
        elif self.pre_proc=="2D":
            self.process2d(N, max_c_size, apply_symmetry, debug)
        else:
            raise Exception(f"cannot deal with pre_proc = {self.pre_proc}")

        if debug is True:
            t2 = time.time()
            print("done, time elapsed: %.2f sec" % (t2-t1))   
        self.save_data()

    @h5_file_access
    def process_qxy(self, qmax, dq, det_ext, apply_symmetry=True,
                    N=8, max_c_size=1024, debug=True):
        """ produce qx-qy maps corresponding to each exposure
            NOTE: it is not feasible have the qx/qy coordinates to span a wide range, due to the very
                  different sample-to-detector distances for SAXS and WAXS. Instead, produce multiple
                  qx-qy maps with different q-resolution if needed
            qgrid is -qmax,dq,qmax, accept a single detector if det_ext is specified
            save the maps under qxqy/{det_ext}, or qxqy/merged if det_ext is None
        """
        if det_ext is None:
            detectors = self.detectors
            map_key = "merged"
        else:
            detectors = [d for d in self.detectors if d.extension==det_ext]
            map_key = det_ext
        qN = 1+2*int(qmax/dq+0.5)
        qgrid = np.linspace(-qmax, qmax, qN)   # qgrid is evenly spaced at dq and contains q=0
        qmax += dq/2                           # bin edges is half a bin wdith outside of the max value
        qrange = [-qmax, qmax]
        
        # prepare the info needed for processing
        dQx = {}
        dQy = {}
        dMask = {}
        dQxyMask = {}
        dWeight = {}
        
        for i in range(len(detectors)):
            exp = detectors[i].exp_para
            dMask[i] = ~unflip_array(exp.mask.map, exp.flip)
            dQx[i] = unflip_array(exp.xQ, exp.flip)[dMask[i]]
            dQy[i] = unflip_array(exp.yQ, exp.flip)[dMask[i]]
            ones = np.ones_like(dQx[i])
            dWeight[i] = fh.histogram2d(dQx[i], dQy[i], bins=(qN, qN), range=(qrange, qrange), weights=ones) 
            dWeight[i] *= detectors[i].fix_scale
            dQxyMask[i] = (dWeight[i]>0)
        parms = [dQx,dQy,dMask,dQxyMask,dWeight]
        
        # for corrections: polarization and solid angle 
        #QPhiCorF = np.ones_like()

        pool = mp.Pool(N)
        jobs = []
        results = {}   # scanning data sets are too large, process one sample at a time
        for sn,dh5 in self.h5xs.items():
            s = dh5.dshape(dh5.det_name[detectors[0].extension])
            if len(s)!=3:
                raise Exception("don't know how to handle shape: ", s)
            results[sn] = {}
            img_grps = [dh5.dset(dh5.det_name[det.extension], get_path=True, sn=sn) for det in detectors]
            for frn_start in range(0, s[0], max_c_size):
                frn_stop = min(frn_start+max_c_size, s[0])
                frns = range(frn_start, frn_stop)
                job = pool.map_async(proc_merge_qxy, [(dh5.fn, img_grps, sn, frns, debug, parms, qN, qrange)])  
                jobs.append(job)
                
        pool.close()
        for job in jobs:
            [sn, frn0, data] = job.get()[0]
            results[sn][frn0] = data
            print(f"data received: sn={sn}                \r", end="")
        pool.join()

        dd2 = MatrixWithCoords()
        dd2.xc = qgrid
        dd2.xc_label = "qx"
        dd2.yc = qgrid
        dd2.yc_label = "qy"
        dd2.err = None
        
        arr_data = {}
        # this is needed to apply symmatry
        sm = np.sum(np.asarray([m for m in dQxyMask.values()], dtype=np.int8), axis=0)
        sm += np.fliplr(np.flipud(sm))

        self.enable_write(True)
        for sn in results.keys():  
            arr_ds = np.vstack([results[sn][frn0] for frn0 in sorted(results[sn].keys())])
            if apply_symmetry:
                arr_ds[np.isnan(arr_ds)] = 0
                arr_ds += np.flip(np.flip(arr_ds, axis=2), axis=1)
                arr_ds /= sm
            dd2.d = arr_ds[0]
            self.pack(sn, 'qxy', map_key, arr_data={'d': arr_ds}, data_proto=dd2)
        self.enable_write(False)
        
    @h5_file_access
    def process1d0(self, N=8, max_c_size=1024, debug=True):
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

            s = dh5.dshape(dh5.det_name[self.detectors[0].extension])
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
                
                dh5.explicit_open_h5()
                for idx, x in np.ndenumerate(t):  # idx should enumerate the outter-most indices
                    images = {det.extension:
                              dh5.dset(dh5.det_name[det.extension])[idx][i*c_size:i*c_size+nframes] for det in detectors}
                    if N>1: # multi-processing, need to keep track of total number of active processes
                        job = pool.map_async(proc_merge1d0, [(images, sn, nframes, fcnt, 
                                                             debug, detectors, self.qgrid)])
                        jobs.append(job)
                    else: # serial processing
                        [sn, fr1, data] = proc_merge1d0((images, sn, nframes, fcnt,
                                                        debug, detectors, self.qgrid))
                        results[sn][fr1] = data
                    fcnt += nframes
                dh5.explicit_close_h5()
                
        if N>1:             
            for job in jobs:
                [sn, fr1, data] = job.get()[0]
                results[sn][fr1] = data
                print(f"data received: sn={sn}, fr1={fr1}                    ")
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

    @h5_file_access
    def process1d(self, N=8, max_c_size=1024, debug=True):
        """ produce merged I(q) profiles
            the bottleneck is reading the data
        """
        qgrid = self.qgrid 
        detectors = self.detectors
        q_bin_ranges = get_bin_ranges_from_grid(qgrid)
        bin_ranges = [q_bin_ranges,len(qgrid)] 
        
        print("this might take a while ...")
        
        # prepare the info needed for processing
        dQ = {}
        dMask = {}
        dWeight = {}
        scale = {}
        
        for i in range(len(detectors)):
            exp = detectors[i].exp_para
            dMask[i] = ~unflip_array(exp.mask.map, exp.flip)
            dQ[i] = unflip_array(exp.Q, exp.flip)[dMask[i]]
            ones = np.ones_like(dQ[i])
            dWeight[i] = np.hstack([fh.histogram1d(dQ[i], bins=qN, range=qrange, weights=ones) for qrange,qN in q_bin_ranges])
            scale[i] = detectors[i].fix_scale
        cQMask = ~(np.sum([dWeight[i] for i in range(len(detectors))], axis=0)>0)

        parms = [dQ,dMask,cQMask,dWeight,scale]        
        # for corrections: polarization and solid angle 
        #QPhiCorF = np.ones_like()

        pool = mp.Pool(N)
        jobs = []
        results = {}   # scanning data sets are too large, process one sample at a time
        errbars = {}
        for sn,dh5 in self.h5xs.items():
            s = dh5.dshape(dh5.det_name[detectors[0].extension])
            if len(s)!=3:
                raise Exception("don't know how to handle shape: ", s)
            results[sn] = {}
            errbars[sn] = {}
            img_grps = [dh5.dset(dh5.det_name[det.extension], get_path=True, sn=sn) for det in detectors]  
            for frn_start in range(0, s[0], max_c_size):
                frn_stop = min(frn_start+max_c_size, s[0])
                frns = range(frn_start, frn_stop)
                job = pool.map_async(proc_merge1d, [(dh5.fn, img_grps, sn, frns, debug, parms, bin_ranges)])  
                jobs.append(job)
            #img_grps = [dh5.dset(dh5.det_name[det.extension], get_path=True, sn=sn) for det in detectors]  
            #job = pool.map_async(proc_merge1d, [(dh5.fn, img_grps, sn, debug, parms, bin_ranges)])  
            #jobs.append(job)
                
        pool.close()
        for job in jobs:
            [sn, frn0, data, err] = job.get()[0]
            results[sn][frn0] = data
            errbars[sn][frn0] = err
            print(f"data received: sn={sn}                \r", end="")
        pool.join()

        for sn in results.keys():
            data = []
            for frn0 in sorted(results[sn].keys()):
                for i in range(len(results[sn][frn0])):
                    d1 = Data1d()
                    d1.qgrid = qgrid
                    d1.data = results[sn][frn0][i]
                    d1.err = errbars[sn][frn0][i]
                    data.append(d1)
            self.add_proc_data(sn, 'azi_avg', 'merged', data)            
            

    @h5_file_access
    def process2d(self, N=8, max_c_size=1024, apply_symmetry=False, debug=True):
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
   
        pool = mp.Pool(N)
        jobs = []
        results = {}   # scanning data sets are too large, process one sample at a time
        for sn,dh5 in self.h5xs.items():
            s = dh5.dshape(dh5.det_name[detectors[0].extension])
            if len(s)!=3:
                raise Exception("don't know how to handle shape: ", s)
            results[sn] = {}
            img_grps = [dh5.dset(dh5.det_name[det.extension], get_path=True, sn=sn) for det in detectors]
            for frn_start in range(0, s[0], max_c_size):
                frn_stop = min(frn_start+max_c_size, s[0])
                frns = range(frn_start, frn_stop)
                job = pool.map_async(proc_merge2d, [(dh5.fn, img_grps, sn, frns, debug, parms, bin_ranges)])  
                jobs.append(job)
                
        pool.close()
        for job in jobs:
            [sn, frn0, data] = job.get()[0]
            results[sn][frn0] = data
            print(f"data received: sn={sn}                \r", end="")
        pool.join()

        arr_data = {}
        # this is needed to apply symmatry
        sm = np.sum(np.asarray([m for m in dQPhiMask.values()], dtype=np.int8), axis=0)
        Np = int(sm.shape[0]/2)
        sm += np.vstack([sm[Np:,:], sm[:Np,:]])
        dd2 = MatrixWithCoords()
        dd2.xc = qgrid
        dd2.xc_label = "q"
        dd2.yc = phigrid
        dd2.yc_label = "phi"
        dd2.err = None
        
        self.enable_write(True)
        for sn in results.keys():  
            arr_ds = np.vstack([results[sn][frn0] for frn0 in sorted(results[sn].keys())])
            if apply_symmetry:
                arr_ds[np.isnan(arr_ds)] = 0
                arr_ds += np.hstack([arr_ds[:,Np:,:], arr_ds[:,:Np,:]])
                arr_ds /= sm
            dd2.d = arr_ds[0]
            self.pack(sn, 'qphi', 'merged', arr_data={'d': arr_ds}, data_proto=dd2)
        self.enable_write(False)
            
    @h5_file_access
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
            s = dh5.dshape(dh5.det_name[self.detectors[0].extension])
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

    def export_data(self, fn, data_keys=None, path='.'):
        """ export all data under "overall" if there are multiple samples
        """
        if len(self.h5xs)==1:
            sn = self.samples[0]
        elif 'overall' not in self.proc_data.keys():
            raise Exception(f"Expecting overall data but not found ...")
        else:
            sn = "overall"
        
        fh5 = h5py.File(f"{path}/{fn}", "w-")  # fail if the file exists
        if data_keys is None:
            data_keys = self.proc_data[sn].keys()
        for data_key in data_keys:
            sks = self.proc_data[sn][data_key].keys()
            for sub_key in sks:
                print(f"{sn}, {data_key}, {sub_key}        \r", end="")
                self.pack(sn, data_key, sub_key, fh5=fh5)
        fh5.close()
    
    @h5_file_access
    def save_data(self, save_sns=None, save_data_keys=None, save_sub_keys=None, quiet=False):
        print("saving processed data ...")
        if save_sns is None:
            save_sns = self.samples
            if "overall" in self.proc_data.keys():
                save_sns = self.samples+["overall"]
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
        if not quiet:
            print("done.                      ")
    
    @h5_file_access
    def pack(self, sn, data_key, sub_key, fh5=None, arr_data=None, data_proto=None):
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
            processed data can be provided as arr_data and data_proto, e.g. when data are a stack of MatrixWithCoords 
                arr_data should be a dictionary corresponding to the attributes of data_proto
                data_proto should be an instance of the data of the same class, with values filled (e.g. including MatrixWithCoords.d)
        """
        if arr_data is None:
            data = self.proc_data[sn][data_key][sub_key]
            if isinstance(data, list):
                d0 = data[0]
                n = len(data)
            else:
                d0 = data
                n = 1
        else:
            k = list(arr_data.keys())[0]
            n = arr_data[k].shape[0]
            d0 = data_proto
            
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
                raise Exception(f"data type of {data_key} for {sn} ({dtype}) does not match existing data ({grp.attrs['type']})")
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
                    if grp[sub_key].shape!=sd.shape:
                        del grp[sub_key]
                        #fh5.flush()
                        grp.create_dataset(sub_key, data=sd)
                    else:
                        grp[sub_key][...] = sd
                else:
                    grp.create_dataset(sub_key, data=sd)
                return

        if not sub_key in list(grp.keys()):
            grp.create_group(sub_key)
        grp = grp[sub_key]
        grp.attrs['len'] = n
        for k in save_fields[dtype]['unique']:
            if not k in d0.__dict__.keys():
                continue
            if d0.__dict__[k] is None:  # e.g. err in MatrixWithCoords
                continue
            if n==1:
                sd = d0.__dict__[k]
            else:
                if arr_data is None:
                    sd = np.array([d.__dict__[k] for d in data])
                else:
                    sd = arr_data[k]
            if k in grp.keys():
                grp[k][...] = sd
            else:
                grp.create_dataset(k, data=sd)
    
    @h5_file_access
    def load_data(self, samples=None, read_data_keys=None, read_sub_keys=None, quiet=False):
        if samples is None:
            samples = self.fh5.keys()
        elif isinstance(samples, str):
            samples = [samples]
        for sn in samples:
            if not sn in self.proc_data.keys():
                if not quiet:
                    print(f"loading data for {sn}")
                self.proc_data[sn] = {}
            if read_data_keys is None:
                dks = list(self.fh5[sn].keys())
            else:
                dks = list(set(read_data_keys) & set(self.fh5[sn].keys()))
            for data_key in dks:
                dtype = self.fh5[sn][data_key].attrs['type']
                if not data_key in self.proc_data[sn].keys():
                    self.proc_data[sn][data_key] = {}
                if dtype=='Data1d':
                    d0 = Data1d()
                elif dtype=="MatrixWithCoords":
                    d0 = MatrixWithCoords()
                for field in save_fields[dtype]['shared']:
                    d0.__dict__[field] = self.fh5[sn][data_key].attrs[field]

                if read_sub_keys is None:
                    sks = list(self.fh5[sn][data_key].keys())
                else:
                    sks = list(set(read_sub_keys) & set(self.fh5[sn][data_key].keys()))
                for sub_key in sks:
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
        if not quiet:
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