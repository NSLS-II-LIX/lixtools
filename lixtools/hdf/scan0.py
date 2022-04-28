from py4xs.data2d import Data2d,MatrixWithCoords
from py4xs.slnxs import Data1d
from py4xs.hdf import lsh5,h5xs
import numpy as np
import multiprocessing as mp
import json,os,copy

save_fields = {"Data1d": {"shared": ['qgrid'], "unique": ["data", "err", "trans", "trans_e", "trans_w"]},
               "MatrixWithCoords": {"shared": ["xc", "yc", "xc_label", "yc_label"], "unique": ["d", "err"]},
               "ndarray": {"shared": [], "unique": []}
              }


def mp_proc(data):
    pass

def merge_d1s(d1s, avg_by_err=False):
    s0 = Data1d()
    s0.qgrid = d1s[0].qgrid
    d_tot = np.zeros(s0.qgrid.shape)
    d_max = np.zeros(s0.qgrid.shape)
    d_min = np.zeros(s0.qgrid.shape)+1.e32
    e_tot = np.zeros(s0.qgrid.shape)
    c_tot = np.zeros(s0.qgrid.shape)
    w_tot = np.zeros(s0.qgrid.shape)
    
    for d1 in d1s:        
        # empty part of the data is nan
        idx = ~np.isnan(d1.data)
        c_tot[idx] += 1
        if avg_by_err:
            wt = 1/d1.err[idx]**2
            d_tot[idx] += wt*d1.data[idx]
            e_tot[idx] += d1.err[idx]**2*wt**2
            w_tot[idx] += wt
        else: # simple averaging
            d_tot[idx] += d1.data[idx]
            e_tot[idx] += d1.err[idx]
        
        idx1 = (np.ma.fix_invalid(d1.data, fill_value=-1)>d_max).data
        d_max[idx1] = d1.data[idx1]
        idx2 = (np.ma.fix_invalid(d1.data, fill_value=1e32)<d_min).data
        d_min[idx2] = d1.data[idx2]
                
    if avg_by_err:
        s0.data = d_tot/w_tot
        s0.err = np.sqrt(e_tot)/w_tot
    else: # simple averaging
        idx = (c_tot>0)
        s0.data = d_tot
        s0.err = e_tot
        s0.data[idx] /= c_tot[idx]
        s0.err[idx] /= np.sqrt(c_tot[idx])
        s0.data[~idx] = np.nan
        s0.err[~idx] = np.nan

    return s0
        
def proc_merge(args):
    dsets,nframes,starting_frame_no,debug,detectors,qgrid,phigrid = args
        
    ret_d2 = []
    ret_e2 = []
    ndet = len(detectors)
    qms = [None for det in detectors]
        
    if debug is True:
        print(f"processing started: starting frame = #{starting_frame_no}\r", end="")
    for i in range(nframes):
        for j,det in enumerate(detectors):
            img = dsets[j][i]
            d2 = Data2d(img, exp=det.exp_para)
            cf = det.fix_scale*d2.exp.FSA*d2.exp.FPol
            if det.flat is not None:
                cf *= det.flat
            qms[j] = d2.data.conv(qgrid, phigrid, d2.exp.Q, d2.exp.Phi, mask=d2.exp.mask, cor_factor=cf, interpolate='x')
        if ndet>1:
            qm = qms[0].merge(qms[1:])
        else:
            qm = qms[0].d
        ret_d2.append(qm.d)
        ret_e2.append(qm.err)

    
    if debug is True:
        print(f"processing completed: {starting_frame_no}.                \r", end="")

    return [starting_frame_no, [ret_d2, ret_e2]]

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

class h5xs_scan(h5xs):
    """ keep the detector information
        import data from raw h5 files, keep track of the file location
        copy the meta data, convert raw data into q-phi maps
        can still show data, 
    """
    def __init__(self, *args, Nphi=32, **kwargs):
        fn = args[0]  # any better way to get this?
        if not os.path.exists(fn):
            open(fn, 'w').close()
        super().__init__(*args, have_raw_data=False, **kwargs)
        self.proc_data = {}
        self.h5xs = {}
        self.Nphi = Nphi
        self.phigrid = np.linspace(-180, 180, self.Nphi)
        # if there are raw data file info, prepare the read only h5xs objects in case needed
        self.samples = list(self.fh5.keys())
        if 'overall' in self.samples:
            self.samples.remove('overall')
            self.attrs['overall'] = {}
        for sn in self.samples:
            self.attrs[sn] = {}
            fn_raw = self.fh5[sn].attrs['source']
            self.h5xs[sn] = h5xs(fn_raw, [self.detectors, self.qgrid], read_only=True)
            self.attrs[sn]['header'] = json.loads(self.fh5[sn].attrs['header'])
            self.attrs[sn]['scan'] = json.loads(self.fh5[sn].attrs['scan'])
            
    def list_data(self):
        for sn,d in self.proc_data.items(): 
            print(sn)
            for dk,dd in d.items(): # data key
                print("++", dk)  # should also print data type
                for sk,sd in dd.items(): # sub key
                    print("++++", sk)  # should also print data size
                    
    def show_data(self, sn, **kwargs):
        self.h5xs[sn].show_data(sn=sn, **kwargs)
        
    def show_data_qphi(self, sn, **kwargs):
        self.h5xs[sn].show_data_qphi(sn=sn, **kwargs)
        
    def show_data_qxy(self, sn, **kwargs):
        self.h5xs[sn].show_data_qxy(sn=sn, **kwargs)
        
    def import_raw_data(self, fn_raw, sn=None, **kwargs):
        """ create new group, copy header, get scan parameters, calculate q-phi map
        """
        dt = h5xs(fn_raw, [self.detectors, self.qgrid], read_only=True)
        if sn is None:
            sn = dt.samples[0]
        elif not sn in dt.samples:
            raise Exception(f"cannot find data on {sn} in {fn_raw}.")
        
        self.h5xs[sn] = dt
        if not sn in self.attrs.keys():
            self.attrs[sn] = {}
        if not sn in self.fh5.keys():
            grp = self.fh5.create_group(sn)
        else:
            grp = self.fh5[sn]
        self.attrs[sn]['source'] = os.path.realpath(fn_raw)
        grp.attrs['source'] = self.attrs[sn]['source'] 
        self.attrs[sn]['header'] = dt.header(sn)
        grp.attrs['header'] = json.dumps(self.attrs[sn]['header'])
        self.attrs[sn]['scan'] = get_scan_parms(dt, sn, **kwargs)
        grp.attrs['scan'] = json.dumps(self.attrs[sn]['scan'])
        
        if not sn in self.proc_data.keys():
            self.proc_data[sn] = {}
        self.list_samples(quiet=True)
        
        return sn
    
    def get_mon(self, sn, *args, **kwargs):
        if not sn in self.h5xs.keys():
            raise Exception(f"no raw data on {sn}.")
        self.h5xs[sn].get_mon(sn, *args, **kwargs)
        if sn not in self.d0s.keys():
            self.d0s[sn] = {}
        if 'attrs' not in self.d0s[sn].keys():
            self.proc_data[sn]['attrs'] = {}
        self.proc_data[sn]['attrs']["transmitted"] = self.h5xs[sn].d0s[sn]["transmitted"]
        self.proc_data[sn]['attrs']["incident"] = self.h5xs[sn].d0s[sn]["incident"]
        self.proc_data[sn]['attrs']["transmission"] = self.h5xs[sn].d0s[sn]["transmission"]
    
    def add_proc_data(self, sn, data_key, sub_key, data):
        if sn not in self.proc_data.keys():
            self.proc_data[sn] = {}
        if data_key not in self.proc_data[sn].keys():
            self.proc_data[sn][data_key] = {}
        self.proc_data[sn][data_key][sub_key] = data
    
    def get_d1s_from_d2s(self, sn, N=8):
        """ calculate azi_avg from qphi
        """ 
        if "qphi" not in self.proc_data[sn].keys():
            raise Exception(f"qphi data do not exist for {sn}.")
            
        
    
    def extract_attr(self, sn, attr_name, func, data_key, sub_key, N=8, **kwargs):
        """ extract an attribute from the pre-processed data using the specified function
            and source of the data (data_key/sub_key)
        """
        data = [func(d, **kwargs) for d in self.proc_data[sn][data_key][sub_key]]
        self.add_proc_data(sn, 'attrs', attr_name, np.array(data))
            
    def process(self, N=8, max_c_size=1024, debug=True, proc_1d=True):
        """ get trans values
            produce azimuthal average and/or q-phi maps
        """
        qgrid = self.qgrid 
        phigrid = self.phigrid
        detectors = self.detectors
        
        print("this might take a while ...")

        for sn in self.h5xs.keys():
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
        
            results = {}
            if N>1:
                pool = mp.Pool(N)
                jobs = []
                
            for i in range(Np):
                if i==Np-1:
                    nframes = n_total_frames - c_size*(Np-1)
                else:
                    nframes = c_size

                if len(s)==3:
                    images = [dh5.dset(dh5.det_name[det.extension])[i*c_size:i*c_size+nframes] for det in detectors]
                    if N>1: # multi-processing, need to keep track of total number of active processes                    
                        job = pool.map_async(proc_merge, [(images, nframes, i*c_size, debug, detectors, qgrid, phigrid)])
                        jobs.append(job)
                    else: # serial processing
                        [fr1, data] = proc_merge((images, nframes, i*c_size, debug, detectors, qgrid, phigrid))
                        results[fr1] = data                
                else: # len(s)==4
                    for j in range(s[0]):  # slow axis
                        images = [dh5.dset(dh5.det_name[det.extension])[j,i*c_size:i*c_size+nframes] for det in detectors]
                        if N>1: # multi-processing, need to keep track of total number of active processes
                            job = pool.map_async(proc_merge, [(images, nframes, i*c_size+j*s[1], debug, detectors, qgrid, phigrid)])
                            jobs.append(job)
                        else: # serial processing
                            [fr1, data] = proc_merge((images, nframes, i*c_size+j*s[1], debug, detectors, qgrid, phigrid)) 
                            results[fr1] = data
                            print(len(data), len(data[0]))

            if N>1:             
                for job in jobs:
                    [fr1, data] = job.get()[0]
                    results[fr1] = data
                    print(f"data received: fr1={fr1}               \r", end="")
                pool.close()
                pool.join()
    
            if not sn in self.proc_data.keys():
                self.proc_data[sn] = {}
            if not "qphi" in self.proc_data[sn].keys():
                self.proc_data[sn]['qphi'] = {}
            self.proc_data[sn]['qphi']['merged'] = []
            #if not "azi_avg" in self.proc_data[sn].keys():
            #    self.proc_data[sn]['azi_avg'] = {}
            #self.proc_data[sn]['azi_avg']['merged'] = []
            for k in sorted(results.keys()):
                for res2d,res2e in zip(*results[k]):
                #for res1d,res1e,res2d,res2e in zip(*results[k]):
                    dd2 = MatrixWithCoords()
                    dd2.xc = qgrid
                    dd2.xc_label = "q"
                    dd2.yc = phigrid
                    dd2.yc_label = "phi"
                    dd2.d = res2d
                    dd2.err = res2e
                    self.proc_data[sn]['qphi']['merged'].append(dd2)

            if debug:
                print(f"done processing {sn}.             ")
    
    def save_data(self, save_sns=None, save_data_keys=None, save_sub_keys=None):
        print("saving processed data ...")
        if save_sns is None:
            save_sns = list(self.fh5.keys())
            if "overall" in self.proc_data.keys():
                save_sns += ["overall"]
        elif not isinstance(save_sns, list):
            save_sns = [save_sns]
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
        print("done.                      ")
    
    def pack(self, sn, data_key, sub_key):
        """ this is for packing processed data, which should be stored under self.proc_data as a dictionary
            sn="overall" is merged from all samples in the h5xs_scan object
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

        fh5 = self.fh5
        # the group fh5[sn] should already exist, created when importing raw data
        # except for "overall"
        if not sn in self.fh5.keys():
            grp = self.fh5.create_group(sn)
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
    
    def load_data(self):
        for sn in self.fh5.keys():
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
        
                                    
    def make_map_from_attr(self, sname, attr_name):
        """ for convenience in data processing, all attributes extracted from the data are saved as
            proc_data[sname]["attrs"][attr_name]
            
            for visiualization and for further data processing (e.g. run tomopy), these attributes need
            to be re-organized (de-snaking, merging) to reflect the shape of the scan/sample view
            
            sn="overall" is reserved for merging data from partial scans
        """

        if sname=="overall":
            samples = self.samples
        elif sname in self.samples:
            samples = [sname]
        else:
            raise exception(f"sample {sn} does not exist") 
        
        maps = []
        for sn in samples:
            if not 'scan' in self.attrs[sn].keys():
                get_scan_parms(self.h5xs[sn], sn)
            if attr_name not in self.proc_data[sn]['attrs'].keys():
                raise Exception(f"attribue {attr_name} cannot be found for {sn}.")
            data = self.proc_data[sn]['attrs'][attr_name].reshape(self.attrs[sn]['scan']['shape'])
            m = MatrixWithCoords()
            m.d = np.copy(data)
            m.xc = self.attrs[sn]['scan']['fast_axis']['pos']
            m.yc = self.attrs[sn]['scan']['slow_axis']['pos']
            m.xc_label = self.attrs[sn]['scan']['fast_axis']['motor']
            m.yc_label = self.attrs[sn]['scan']['slow_axis']['motor']
            if self.attrs[sn]['scan']['snaking']:
                print("de-snaking")
                for i in range(1,self.attrs[sn]['scan']['shape'][0],2):
                    m.d[i] = np.flip(m.d[i])
            if "maps" not in self.proc_data[sn].keys():
                self.proc_data[sn]['maps'] = {}
            maps.append(m)
            self.proc_data[sn]['maps'][attr_name] = m
    
        # assume the scans are of the same type, therefore start from the same direction
        if sname=="overall":
            mm = maps[0].merge(maps[1:])
            if "overall" not in self.proc_data.keys():
                self.proc_data['overall'] = {}
                self.proc_data['overall']['maps'] = {}
            self.proc_data['overall']['maps'][attr_name] = mm