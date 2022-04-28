from py4xs.hdf import h5xs,lsh5
from py4xs.slnxs import trans_mode,estimate_scaling_factor
import numpy as np
import pylab as plt
import json,time,copy

from scipy.interpolate import splrep,sproot,splev,UnivariateSpline

class h5sol_ref(h5xs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "abs_scaling" in self.fh5.attrs:
            self.scaling = json.loads(self.fh5.attrs['abs_scaling'])
            print("Scaling information retrieved from reference data.")
        else:
            self.scaling = {}
        
    def process(self, plot_data=True, timestamp=None, Iabs_w=1.632e-2, **kwargs):
        """ calculate the scaling factor sf needed
            to put Data1d.data on absolute scale: data*sf/trans_w
            Iabs_w is the know absolute water scattering intensity at low q 
            1.632 × 10−2 cm−1 at 293 K, doi:10.1107/S0021889899015216
            temperature-dependent
        """
        sn = self.samples[0]
        if "processed" not in lsh5(self.fh5[sn], top_only=True, silent=True):
            self.load_data()
        else:
            self.load_d1s()
        
        self.set_trans(transMode=trans_mode.external, calc_water_peak=True, trigger="sol", debug='quiet', **kwargs)
        self.average_d1s(filter_data=True)
        
        sdict = {}
        for sn in self.samples:
            if "empty" in sn: # ss should be "empty"
                ss,ts = sn.split('_', maxsplit=1)
                if ss!="empty":
                    raise Exception(f"don't know how to handle {sn}.")
            else: # ss should be water or blank, cn should be top or bottom   
                cn,ss,ts = sn.split('_', maxsplit=2)
                if not ss in["water", "blank"] or cn not in ["top", "bottom"]:
                    raise Exception(f"don't know how to handle {sn}.")

            if not ts in sdict.keys():
                sdict[ts] = {}
            if ss=="empty":
                sdict[ts]["empty"] = self.d1s[sn]["averaged"] 
            else:
                if cn not in sdict[ts].keys():
                    sdict[ts][cn] = {}
                sdict[ts][cn][ss] = self.d1s[sn]["averaged"]
                
        # only keep the data with complete set of information
        bad_ts_list = []
        for ts in sdict.keys():
            if set(sdict[ts].keys())<set(['top', 'bottom', 'empty']):
                bad_ts_list.append(ts)
            elif set(sdict[ts]['top'])<set(['blank', 'water']) or set(sdict[ts]['bottom'])<set(['blank', 'water']):
                bad_ts_list.append(ts)
        for ts in bad_ts_list:
            del sdict[ts]
        if len(sdict)==0:
            raise Exception("No valid data found.")

        if not timestamp in sdict.keys():
            print(f"cannot find the specified timestamp: {timestamp}, taking the average.")
            timestamp = list(sdict.keys())
            
        if plot_data:
            plt.figure()
            ax = plt.gca()
            
        scaling_dict = {}
        for cn in ["top", "bottom"]:
            scaling_dict[cn] = {'trans': 0, 'trans_w': 0, 'Irel': 0}
            n = 0
            for ts in sdict.keys():
                d1w = sdict[ts][cn]['water']
                d1b = sdict[ts][cn]['blank']
                d1w.bkg_cor(sdict[ts]["empty"], inplace=True)
                d1b.bkg_cor(sdict[ts]["empty"], inplace=True)
                d1t = d1w.bkg_cor(d1b)
                if plot_data:
                    ax.plot(d1t.qgrid, d1t.data, label=f"{cn},{ts}")
                avg = np.average(d1t.data[(d1t.qgrid>0.1) & (d1t.qgrid<0.2)])
                tratio = d1w.trans/d1b.trans 
                print(f"{cn},{ts}: trans={d1w.trans:.1f}, trans_ratio={tratio:.3f}, trans_w={d1w.trans_w:.1f}, {avg:.3f}")
                if ts in timestamp:
                    scaling_dict[cn]['trans'] += d1w.trans
                    scaling_dict[cn]['trans_w'] += d1w.trans_w
                    scaling_dict[cn]['Irel'] += avg
                    n += 1
            for k in scaling_dict[cn].keys():
                scaling_dict[cn][k] /= n
            scaling_dict[cn]['sc_factor'] = Iabs_w/scaling_dict[cn]['Irel']
        
        if plot_data:
            plt.legend()
            plt.xlim(0.1, 0.3)
            plt.ylim(0.1, 5)
            plt.show()

        self.fh5.attrs['abs_scaling'] = json.dumps(scaling_dict)
        self.scaling = scaling_dict
        
    def scaling_factor(self, cn=None):
        if self.scaling=={}:
            raise Exception("Scaling info not found, run process() first.")
        if cn in ['top', 'bottom']:
            sc = self.scaling[cn]['sc_factor']
            tw = self.scaling[cn]['trans_w']
        else:
            sc = (self.scaling['top']['sc_factor']+self.scaling['bottom']['sc_factor'])/2
            tw = (self.scaling['top']['trans_w']+self.scaling['bottom']['trans_w'])/2
            
        return sc*tw
    
        
class h5sol_HT(h5xs):
    """ multiple samples, not many frames per sample
    """    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d1b = {}   # buffer data used, could be averaged from multiple samples
        self.buffer_list = {}    
        
    def add_sample(self, db, uid):
        """ add another group to the HDF5 file
            only works at the beamline
        """
        header = db[uid]
        
    def load_d1s(self, sn=None):
        if sn==None:
            samples = self.samples
        elif isinstance(sn, str):
            samples = [sn]
        else:
            samples = sn
        
        for sn in samples:
            super().load_d1s(sn)
            if 'buffer' in list(self.fh5[sn].attrs.keys()):
                self.buffer_list[sn] = self.fh5[sn].attrs['buffer'].split()
        
    def assign_buffer(self, buf_list, debug=False):
        """ buf_list should be a dict:
            {"sample_name": "buffer_name",
             "sample_name": ["buffer1_name", "buffer2_name"],
             ...
            }
            anything missing is considered buffer
        """
        for sn in list(buf_list.keys()):
            if isinstance(buf_list[sn], str):
                self.buffer_list[sn] = [buf_list[sn]]
            else:
                self.buffer_list[sn] = buf_list[sn]
        
        if debug is True:
            print('updating buffer assignments')
        if self.read_only:
            print("h5 file is read-only ...")
            return
        self.enable_write(True)
        for sn in self.samples:
            if sn in list(self.buffer_list.keys()):
                self.fh5[sn].attrs['buffer'] = '  '.join(self.buffer_list[sn])
        self.enable_write(False)

    def change_buffer(self, sample_name, buffer_name):
        """ buffer_name could be just a string (name) or a list of names 
            if sample_name is a list, all samples in the list will be assigned the same buffer
        """
        
        if not isinstance(sample_name, list):
            sample_name = [sample_name]
            
        if isinstance(buffer_name, str):
            buffer_name = [buffer_name]
        for b in buffer_name:
            if b not in self.samples:
                raise Exception(f"invalid buffer name: {b}")

        if self.read_only:
            print("h5 file is read-only ...")
            return
        for sn in sample_name:
            if sn not in self.samples:
                raise Exception(f"invalid sample name: {sn}")
            self.buffer_list[sn] = buffer_name 
            self.enable_write(True)
            self.fh5[sn].attrs['buffer'] = '  '.join(buffer_name)
            self.enable_write(False)
            self.subtract_buffer(sn)
                
    def update_h5(self, debug=False):
        """ raw data are updated using add_sample()
            save sample-buffer assignment
            save processed data
        """
        if debug is True:
            print("updating 1d data and buffer info ...") 
        if self.read_only:
            print("h5 file is read-only ...")
            return
        for sn in self.samples:
            if sn in list(self.buffer_list.keys()):
                self.enable_write(True)
                self.fh5[sn].attrs['buffer'] = '  '.join(self.buffer_list[sn])
                self.enable_write(False)
            self.save_d1s(sn, debug=debug)
        
    def process(self, detectors=None, update_only=False,
                reft=-1, sc_factor=1., save_1d=False, save_merged=False, 
                filter_data=True, debug=False, N = 1):
        """ does everything: load data from 2D images, merge, then subtract buffer scattering
        """
        if filter_data=="keep":
            self.load_d1s()
        self.load_data(update_only=update_only, detectors=detectors, reft=reft, 
                       save_1d=save_1d, save_merged=save_merged, debug=debug, N=N)
        self.set_trans(transMode=trans_mode.from_waxs)
        if filter_data=="keep":
            self.average_samples(update_only=update_only, filter_data=False, selection=None, debug=debug)
        else:
            self.average_samples(update_only=update_only, filter_data=filter_data, debug=debug)
        self.subtract_buffer(update_only=update_only, sc_factor=sc_factor, debug=debug)
        
    def average_samples(self, **kwargs):
        """ if update_only is true: only work on samples that do not have "merged' data
            selection: if None, retrieve from dataset attribute
        """
        super().average_d1s(**kwargs)
            
    def subtract_buffer(self, samples=None, update_only=False, sc_factor=1., debug=False):
        """ if update_only is true: only work on samples that do not have "subtracted' data
            sc_factor: if <0, read from the dataset attribute
        """
        if samples is None:
            samples = list(self.buffer_list.keys())
        elif isinstance(samples, str):
            samples = [samples]
        
        if self.read_only:
            print("h5 file is read-only ...")
            return
        
        if debug is True:
            print("start processing: subtract_buffer()")
            t1 = time.time()
        self.enable_write(True)
        for sn in samples:
            if update_only and 'subtracted' in list(self.d1s[sn].keys()): continue
            if sn not in list(self.buffer_list.keys()): continue
            
            bns = self.buffer_list[sn]
            if isinstance(bns, str):
                self.d1b[sn] = self.d1s[bns]['averaged']  # ideally this should be a link
            else:
                self.d1b[sn] = self.d1s[bns[0]]['averaged'].avg([self.d1s[bn]['averaged'] for bn in bns[1:]], 
                                                                debug=debug)
            if sc_factor=="auto":
                sf = estimate_scaling_factor(self.d1s[sn]['averaged'], self.d1b[sn])
                # Data1d.bkg_cor() normalizes trans first before applying sc_factor
                # in contrast the estimated 
                sf /= self.d1s[sn]['averaged'].trans/self.d1b[sn].trans
                if debug!="quiet":
                    print(f"setting sc_factor for {sn} to {sf:.4f}")
                self.attrs[sn]['sc_factor'] = sf
            elif sc_factor>0:
                self.attrs[sn]['sc_factor'] = sc_factor
                sf = sc_factor
            else:
                sf = self.attrs[sn]['sc_factor']
            self.d1s[sn]['subtracted'] = self.d1s[sn]['averaged'].bkg_cor(self.d1b[sn], 
                                                                          sc_factor=sf, debug=debug)
            self.enable_write(False)

        self.update_h5() 
        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
                
    def plot_sample(self, *args, **kwargs):
        """ show_subtracted:
                work only if sample is background-subtracted, show the subtracted result
            show_subtraction: 
                if True, show sample and boffer when show_subtracted
            show_overlap: 
                also show data in the overlapping range from individual detectors
                only allow if show_subtracted is False
        """
        super().plot_d1s(*args, **kwargs)
     
    def export_txt(self, *args, **kwargs):
        super().export_d1s(*args, **kwargs)
