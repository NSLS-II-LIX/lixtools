from py4xs.data2d import Data2d,MatrixWithCoords
from py4xs.slnxs import Data1d
from py4xs.hdf import lsh5,h5xs
import numpy as np
import multiprocessing as mp
import json,os,pickle

save_fields = {"py4xs.slnxs.Data1d": {"shared": ['qgrid', "transMode"],
                                      "unique": ["data", "err", "trans", "trans_e", "trans_w"]},
               "py4xs.data2d.MatrixWithCoords": {"shared": ["xc", "yc", "xc_label", "yc_label"], 
                                                 "unique": ["d", "err"]},
              }

def proc_merge(args):
    dsets,nframes,starting_frame_no,debug,detectors,qgrid,phigrid = args
        
    ret_d2 = []
    ret_e2 = []
    ret_d1 = []
    ret_e1 = []
    ndet = len(detectors)
    qms = [None for det in detectors]
    if debug is True:
        print(f"processing started: starting frame = #{starting_frame_no}\r", end="")
    for i in range(nframes):
        for j,det in enumerate(detectors):
            img = dsets[j][i]
            d2 = Data2d(img, exp=det.exp_para)
            cf = det.fix_scale*d2.exp.FSA*d2.exp.FPol
            qms[j] = d2.data.conv(qgrid, phigrid, d2.exp.Q, d2.exp.Phi, mask=d2.exp.mask, cor_factor=cf, interpolate='x')
        if ndet>1:
            qm = qms[0].merge(qms[1:])
        else:
            qm = qms[0].d
        #d1,e1 = qm.flatten(axis='x')
        ret_d2.append(qm.d)
        ret_e2.append(qm.err)
        #ret_d1.append(d1)
        #ret_e1.append(e1)
            
    if debug is True:
        print(f"processing completed: {starting_frame_no}.                \r", end="")

    #return [starting_frame_no, ret_d2, ret_e2, ret_d1, ret_e1]
    return [starting_frame_no, [ret_d2, ret_e2]]


def get_scan_parms(dh5xs, sn, prec=0.001):
    """ figure out the scan shape and motor positions, assuming 2D grid scans 
        i.e. sufficient to have a single set of x and y coordinates to specify the location
    """
    shape = dh5xs.header(sn)['shape']
    assert(len(shape)==2)

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
    # for the fast axis, the Newport fly scan sometime repeats position data  
    fpos = dh5xs.fh5[sn][f"primary/data/{motors[1]}"][...].flatten()
    n = int(len(fpos)/shape[0]/shape[1])
    fpos = fpos[::n]         # remove redundancy, specific to fly scanning with Newport
    fpos = fpos[:shape[1]]   # assume these positions are repeating
    fpos = prec*np.floor(fpos/prec)     # slight formatting

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
        self.phigrid = np.linspace(-90, 90, self.Nphi)
        # if there are raw data file info, prepare the read only h5xs objects in case needed
        for sn in self.fh5.keys():
            slef.attrs[sn] = {}
            fn_raw = self.h5xs[sn].attrs['source']
            self.h5xs[sn] = h5xs(fn_raw, [self.detectors, self.qgrid], read_only=True)
            self.attrs[sn]['header'] = json.loads(self.h5xs[sn].attrs['header'])
            self.attrs[sn]['scan'] = json.loads(self.h5xs[sn].attrs['scan'])
            
    def list_data(self):
        for sn,d in self.proc_data.items(): 
            print(sn)
            for dk,dd in d.items(): # data key
                print("++", dk)  # should also print data type
                for sk,sd in dd.items(): # sub key
                    print("++++", sk)  # should also print data size
        
    def import_raw_data(self, fn_raw, sn=None):
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
        self.attrs[sn]['scan'] = get_scan_parms(dt, sn)
        grp.attrs['scan'] = json.dumps(self.attrs[sn]['scan'])
        
        if not sn in self.proc_data.keys():
            self.proc_data[sn] = {}
        
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
            
    def process(self, N=8, max_c_size=1024, debug=True):
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

            if N>1:             
                for job in jobs:
                    [fr1, data] = job.get()[0]
                    results[fr1] = data
                    print(f"data received: fr1={fr1}\r              ", end="")
                pool.close()
                pool.join()
    
            if not sn in self.proc_data.keys():
                self.proc_data[sn] = {}
            if not "qphi" in self.proc_data[sn].keys():
                self.proc_data[sn]['qphi'] = {}
            self.proc_data[sn]['qphi']['merged'] = []
            if not "azi_avg" in self.proc_data[sn].keys():
                self.proc_data[sn]['azi_avg'] = {}
            self.proc_data[sn]['azi_avg']['merged'] = []
            for k in sorted(results.keys()):
                for res in results[k]:
                    dd2 = MatrixWithCoords()
                    dd2.xc = qgrid
                    dd2.xc_label = "q"
                    dd2.yc = phigrid
                    dd2.yc_label = "phi"
                    dd2.d = res[0]
                    dd2.err = res[1]
                    self.proc_data[sn]['qphi']['merged'].append(dd2)
                    dd1 = Data1d()
                    dd1d,dd1e = dd2.flatten(axis='x')
                    dd1.qgrid = dd2.xc
                    dd1.data = dd1d
                    dd1.err = dd1e
                    self.proc_data[sn]['azi_avg']['merged'].append(dd1)

            if debug:
                print(f"done processing {sn}.")
    
    def pack(self, sn, data_key, sub_key):
        """ this is for packing processed data, which should be stored under self.proc_data as a dictionary
            the key is the name/identifier of the processed data, e.g. "qphi", "azi_avg"
            the sub_key may not always be necessary, but is required for consistency 
            all sub_keys should be lists/instances of Data1d or MatrixWithCoords, with the same "shared" properties
                e.g. proc_data["maps"]["trans"], proc_data["azi_avg"]["subtracted"] 

            pack_data() saves the data into the h5 file under the group processed/{data_key}/{sub_key}

        """
        n = len(self.proc_data[sn][data_key][sub_key])
        if n==1:
            d0 = self.proc_data[sn][data_key][sub_key]
        else:
            d0 = self.proc_data[sn][data_key][sub_key][0]
        if not d0.__class__ in save_fields.keys():
            raise Exception(f"{d0.__class__} is not supported for packing.")

        fh5 = self.fh5
        # the sample group should have been created when importing raw data
        grp = fh5[sn]
        
        # if the data group exists, the saved attributes needs to match those for the new data, 
        # otherwise raise exception, in case there is a conflict with existing data
        # the "shared" fields of the data, e.g. qgrid in azimuthal average, are then saved as attributes of the data group
        if data_key in list(grp.keys()):
            grp = grp[data_keys]
            for k in save_fields[d0.__class__]["shared"]:
                if not np.equal(d0.__dict__[k], grp.attr[k]):
                    raise Exception(f"{k} in {data_key} for {sn} does not match existing data")
        else:
            grp = grp.create_group(data_key)
            for k in save_fields[d0.__class__]["shared"]:
                grp.attrs[k] = d0.__dict__[k]

        # under the group, save the "unique" fields as datasets, after reorganizing if necessary (always a list?)
        # the data group should be named as sub_key.unique_field
        # chunk size??



        if grp[g0][0].shape[1]!=len(self.qgrid): # if grp[g0].value[0].shape[1]!=len(self.qgrid):
            # new size for the data
            del fh5[sn+'/processed']

        grp = fh5[f"{sn}/processed"]
        g0 = lsh5(grp, top_only=True, silent=True)[0]


        data = self.proc_data[data_key]
        if isinstance(data, np.ndarray):
            # save directly, no other information necessary
            pass

        if not isinstance(data, list):
            d0 = data
            data = [d0]
        else:
            d0 = data[0]
        if not data.__class__ in save_fields.keys():
            raise Exception(f"don't know how to pack {data}") 

        for k in save_fields:
            grp = fh5[sn+'/processed']
            g0 = lsh5(grp, top_only=True, silent=True)[0]
            if grp[g0][0].shape[1]!=len(self.qgrid): # if grp[g0].value[0].shape[1]!=len(self.qgrid):
                # new size for the data
                del fh5[sn+'/processed']
                grp = fh5[sn].create_group("processed")        

    def unpack(self, data_key):
        pass


                            
    def load_qphi(self):
        pass
        
    def make_map_from_attr(self, data, sn, key):
        """ for convenience in data processing, all processed data are saved as 1d lists'
            e.g. d0s[sn]["transmitted"], d1s[sn]["azi_avg"], d2s[sn]["qphi"]
            
            for visiualization and for further data processing (e.g. run tomopy), d0s need
            to be re-organized to reflect the shape of the scan
            the result is saved into d2s[sn]["maps"][key]
        """
        if not 'scan' in self.attrs[sn].keys():
            get_scan_parms(self.h5xs[sn], sn)
        data = data.reshape(self.attrs[sn]['scan']['shape'])
        if not sn in self.proc_data.keys():
            self.proc_data[sn] = {}
        m = MatrixWithCoords()
        m.d = data
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
        self.proc_data[sn]['maps'][key] = m
    