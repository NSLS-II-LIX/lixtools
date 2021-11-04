from py4xs.data2d import MatrixWithCoords
from py4xs.hdf import lsh5,h5xs
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from py4xs.local import incident_monitor_field,transmitted_monitor_field

def integrate_mon(em, ts, ts0, exp):
    """ integrate monitor counts
        monitor counts are given by em with timestamps ts
        ts0 is the timestamps on the exposures, with duration of exp
        
        assume ts and ts0 are 1d arrays
    """
    ffe = interp1d(ts, em)
    em0 = []
    for t in ts0:
        tt = np.concatenate(([t], ts[(ts>t) & (ts<t+exp)], [t+exp]))
        ee = ffe(tt)
        em0.append(simpson(ee, tt))
    return np.asarray(em0)/exp

def get_monitor_counts(grp, fieldName):
    """ look under a data group (grp) that belong to a specific sample, find the stream that contains fieldName
        caluclate the monitor counts based on the given timestamps (ts) and exposure time
    """
    strn = None
    for stream in list(grp):
        if not 'data' in list(grp[stream]):
            continue
        if fieldName in list(grp[stream]["data"]):
            strn = stream
            break
    if strn is None:
        raise Exeption(f"could not find the stream that contains {fieldName}.")
    
    data = grp[strn]["data"][fieldName][...]
    ts = grp[strn]["timestamps"][fieldName][...]

    return strn,ts,data

save_fields = {"py4xs.slnxs.Data1d": {"shared": ['qgrid', "transMode"],
                                      "unique": ["data", "err", "trans", "trans_e", "trans_w"]},
               "py4xs.data2d.MatrixWithCoords": {"shared": ["xc", "yc", "xc_label", "yc_label"], 
                                                 "unique": ["d", "err"]},
              }

class h5xs_scan(h5xs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proc_data = {}
        
    def proc_data():
        """ get trans values
            produce azimuthal average and/or q-phi maps
        """
    
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
        # make sure the processed group exisits
        if not "processed" in list(fh5[sn].keys()):
            grp = fh5[sn].create_group("processed")
        else:
            grp = fh5[f"{sn}/processed"]

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

    def get_scan_parms(self, sn, prec=0.001):
        """ figure out the scan shape and motor positions, assuming 2D grid scans 
            i.e. sufficient to have a single set of x and y coordinates to specify the location
        """
        shape = self.header(sn)['shape']
        assert(len(shape)==2)
        
        motors = self.header(sn)['motors']
        pn = self.header(sn)['plan_name']
        if pn=="raster":
            snaking = True
        elif "snaking" in self.header(sn).keys():
            snaking = self.header(sn)["snaking"][-1]
        else: 
            snaking = False
            
        if len(motors)!=2:
            raise Exception(f"expecting two motors, got {motors}.")
        # slow axis is the first motor 
        spos = self.fh5[sn][f"primary/data/{motors[0]}"][...] 
        # for the fast axis, the Newport fly scan sometime repeats position data  
        fpos = self.fh5[sn][f"primary/data/{motors[1]}"]
        n = int(len(fpos)/shape[0]/shape[1])
        fpos = fpos[::n]         # remove redundancy, specific to fly scanning with Newport
        fpos = fpos[:shape[1]]   # assume these positions are repeating
        fpos = prec*np.floor(fpos/prec)     # slight formatting
        
        if not sn in self.attrs.keys():
            self.attrs[sn] = {}
        self.attrs[sn]['scan'] = {"shape": shape,
                                  "snaking": snaking, 
                                  "fast_axis": {"motor": motors[1], "pos": fpos}, 
                                  "slow_axis": {"motor": motors[0], "pos": spos}}
        
        
    def get_mon(self, sn=None, trigger=None, gf_sigma=2, exp=1, 
                force_synch=-0.25, force_synch_trig=0, debug=False, plot_trigger=False, **kwargs): 
        """ calculate the monitor counts for each data point
            1. if the monitors are read together with the detectors 
            2. if the monitors are used asynchronously, monitor values would need to be integated 
               based on timestampls; a trigger must be provided 
                2a: in fly scans, this should be a motor name
                2b: for solution scattering, use "sol" as trigger

            the timestamps on the trigger and em1/em2 may not be synced, dt0 provides a correction 
            to the trigger timestamp
            
            if plot_trigger=True and a single sample is named as sn, generate a plot for verification

            timestamps on em1 appear to be way off, ntpd not running
            if force_synch is non-zero, use first em2 timestamp + force_synch as start of em1 
        """
        if sn is None or isinstance(sn, list):
            if plot_trigger:
                plot_trigger = False           # plot for a single sample only
                print("Disabling plot_trigger since not a single sample name is specified.")
            samples = self.samples
        else:
            samples = [sn]
           
        for s in samples:
            if "pilatus" in self.header(s).keys():
                md = self.header(s)['pilatus']
                if 'exposure_time' in md.keys():
                    exp=md['exposure_time']

            if trigger is None:
                #raise Exception("the motor that triggers data collection must be specified.")
                print("monitors are used as detectors.")
            elif trigger=="sol":
                dn = list(self.det_name.values())[0]
                # expect a finite but minimal offset in time since all come from the same IOC server
                ts0 = self.fh5[f'{s}/primary/timestamps/{dn}'][...].flatten()
                dshape = self.fh5[f"{s}/primary/data/{dn}"].shape[0]   # length of the time sequence
                if len(ts0)==1: # multiple exposures, single trigger, as in HT measurements
                    ts0 = ts0[0]+np.arange(dshape)*exp    
            elif trigger in self.fh5[f'{s}/primary/timestamps'].keys():
                ts0 = self.fh5[f'{s}/primary/timestamps/{trigger}'][...].flatten()
                dshape = self.fh5[f"{s}/primary/data/{list(self.det_name.values())[0]}"].shape[:-2]
                if len(dshape)>1:
                    if len(dshape)>2:
                        raise Exception(f"Don't know how to handle data shape {dshape}")
                    dshape = dshape[0]*dshape[1]
                else:
                    dshape = dshape[0]
                if len(ts0) != dshape:
                    raise Exception(f"mistached timestamp length: {len(ts0)} vs {dshape}")
                if len(ts0)>1: # expect the monitor data to be 1D
                    ts0 = ts0.flatten()
            else:
                raise Exception(f"timestamp data for {trigger} cannot be found.")

            strn,ts2,trans_data = get_monitor_counts(self.fh5[sn], transmitted_monitor_field)
            strn,ts1,incid_data = get_monitor_counts(self.fh5[sn], incident_monitor_field)
            if force_synch!=0: # timestamps between em1/em2 
                ts1 = ts1-ts1[0]+ts2[0]+force_synch

            if strn=="primary":
                trans_data0 = trans_data
                incid_data0 = incid_data
            else:
                trans_data0 = integrate_mon(trans_data, ts2, ts0+force_synch_trig, exp)
                incid_data0 = integrate_mon(incid_data, ts1, ts0+force_synch_trig, exp)                

                if plot_trigger:
                    plt.figure()
                    plt.plot(ts2, trans_data/np.max(trans_data))
                    plt.plot(ts0, trans_data0/np.max(trans_data), "o")
                    plt.plot(ts1, incid_data/np.max(incid_data))
                    plt.plot(ts0, incid_data0/np.max(incid_data), "o")

            if not hasattr(self, "d0s"):
                self.d0s = {}
            if not sn in self.d0s.keys():
                self.d0s[s] = {}
            self.d0s[s]["transmitted"] = trans_data0
            self.d0s[s]["incident"] = incid_data0
            transmission = trans_data0/incid_data0
            transmission /= np.nanmean(transmission)
            self.d0s[s]["transmission"] = transmission
            
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
            self.get_scan_parms(sn)
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
        self.proc_data[sn][key] = m
    