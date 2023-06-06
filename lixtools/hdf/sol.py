from py4xs.hdf import h5xs,lsh5,h5_file_access,get_d1s_from_grp  
from py4xs.slnxs import trans_mode,estimate_scaling_factor
import numpy as np
import pylab as plt
import json,time,copy
import h5py

from scipy.interpolate import splrep,sproot,splev,UnivariateSpline

grp_empty_cells = ".empty"

class h5sol_ref(h5xs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_scaling()

    @h5_file_access  
    def check_scaling(self):
        if "abs_scaling" in self.fh5.attrs:
            self.scaling = json.loads(self.fh5.attrs['abs_scaling'])
            print("Scaling information retrieved from reference data.")
        else:
            self.scaling = {}
        
    @h5_file_access  
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

        self.enable_write(True)
        self.fh5.attrs['abs_scaling'] = json.dumps(scaling_dict)
        self.enable_write(False)
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
        
    def load_d1s(self, sn=None, read_attrs=['buffer']):
        super().load_d1s(sn, read_attrs=read_attrs)
        
    @h5_file_access  
    def assign_sample_attr(self, sa_dict, attr_name="buffer", debug=False):
        attr_list = self.__dict__[f"{attr_name}_list"]
        for sn in list(sa_dict.keys()):
            if isinstance(sa_dict[sn], str):
                attr_list[sn] = [sa_dict[sn]]
            else:
                attr_list[sn] = sa_dict[sn]
        
        if debug is True:
            print('updating buffer assignments')
        if self.read_only:
            print("h5 file is read-only ...")
            return
        self.enable_write(True)
        for sn in self.samples:
            if sn in list(attr_list.keys()):
                self.fh5[sn].attrs[attr_name] = '  '.join(attr_list[sn])
            elif attr_name in list(self.fh5[sn].attrs):
                del self.fh5[sn].attrs[attr_name]
        self.enable_write(False)
        
    @h5_file_access  
    def assign_buffer(self, buf_list, debug=False):
        """ buf_list should be a dict:
            {"sample_name": "buffer_name",
             "sample_name": ["buffer1_name", "buffer2_name"],
             ...
            }
            anything missing is considered buffer
        """
        self.assign_sample_attr(buf_list, "buffer")

    @h5_file_access  
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
                
    @h5_file_access  
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
            self.enable_write(True)
            self.fh5[sn].attrs['selected'] = self.attrs[sn]['selected']
            if sn in list(self.buffer_list.keys()):
                self.fh5[sn].attrs['buffer'] = '  '.join(self.buffer_list[sn])
                self.fh5[sn].attrs['sc_factor'] = self.attrs[sn]['sc_factor']
            elif 'buffer' in list(self.fh5[sn].attrs):
                del self.fh5[sn].attrs['buffer']
            self.enable_write(False)                
            self.save_d1s(sn, debug=debug)
        
    def process(self, detectors=None, update_only=False,
                reft=-1, sc_factor=1., save_1d=False, save_merged=False, 
                filter_data=True, selection=None, debug=False, N = 1):
        """ does everything: load data from 2D images, merge, then subtract buffer scattering
        """
        if filter_data=="keep":
            self.load_d1s()
        self.load_data(update_only=update_only, detectors=detectors, reft=reft, 
                       save_1d=save_1d, save_merged=save_merged, debug=debug, N=N)
        self.set_trans(transMode=trans_mode.from_waxs)
        if filter_data=="keep":
            self.average_d1s(update_only=update_only, filter_data=False, selection=None, debug=debug)
        else:
            self.average_d1s(update_only=update_only, filter_data=filter_data, selection=selection, debug=debug)
        self.subtract_buffer(update_only=update_only, sc_factor=sc_factor, debug=debug)
                    
    def subtract_buffer(self, samples=None, update_only=False, sc_factor=1., 
                        input_grp='averaged', debug=False):
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
        #self.enable_write(True)
        for sn in samples:
            if update_only and 'subtracted' in list(self.d1s[sn].keys()): continue
            if sn not in list(self.buffer_list.keys()): continue
            
            bns = self.buffer_list[sn]
            if isinstance(bns, str):
                self.d1b[sn] = self.d1s[bns][input_grp]  # ideally this should be a link
            else:
                self.d1b[sn] = self.d1s[bns[0]][input_grp].avg([self.d1s[bn][input_grp] for bn in bns[1:]], 
                                                                debug=debug)
            if sc_factor=="auto":
                sf = "auto" #estimate_scaling_factor(self.d1s[sn][input_grp], self.d1b[sn])
                # Data1d.bkg_cor() normalizes trans first before applying sc_factor
                # in contrast the estimated 
                #sf /= self.d1s[sn][input_grp].trans/self.d1b[sn].trans
                #if debug!="quiet":
                #    print(f"setting sc_factor for {sn} to {sf:.4f}")
                self.attrs[sn]['sc_factor'] = sf
            elif sc_factor>0:
                self.attrs[sn]['sc_factor'] = sc_factor
                sf = sc_factor
            else:
                sf = self.attrs[sn]['sc_factor']
            self.d1s[sn]['subtracted'] = self.d1s[sn][input_grp].bkg_cor(self.d1b[sn], sc_factor=sf,
                                                                         label=f'{sn}-subtracted', debug=debug)
        #self.enable_write(False)

        self.update_h5() 
        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
                
    def plot_d1s(self, sn, show_subtracted=True, **kwargs):
        if show_subtracted:
            src_d1 = self.d1s[sn]['averaged']
            bn = self.buffer_list[sn][0]  # this does not accomordate multiple buffers
            bkg_d1 = self.d1s[bn]['averaged']
            show_subtracted='subtracted'
        else:
            src_d1 = None
            bkg_d1 = None
        super().plot_d1s(sn, show_subtracted=show_subtracted, src_d1=src_d1, bkg_d1=bkg_d1, **kwargs)
     
    def export_d1s(self, samples=None, exclude_buf=True, **kwargs):
        """ exclude_buf is only considered when samples is None
        """
        if samples is None: 
            if exclude_buf:
                samples = list(self.buffer_list.keys())
            else:
                samples = self.samples
        elif isinstance(samples, str):
            samples = [samples]
        super().export_d1s(samples=samples, **kwargs)

        
class h5sol_fc(h5sol_HT):
    """ 
    fixed cells, each sample has a corresponding buffer, which could be from a different holder
    each sample/buffer also has a corresponding empty cell, from a different holder
    new key under self.d1s[sn]: empty_subtracted
    
    before processing starts, make soft links to the data file that contain empty cell scattering
    assign empty cell scattering to each sample
    
    """    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, exclude_sample_names=['.empty'], **kwargs)
        self.empty_list = {}
        self.exclude_sample_names = [grp_empty_cells]
        self.list_samples(quiet=True)

    def load_d1s(self, sn=None):
        super().load_d1s(sn=sn, read_attrs=['buffer', 'empty'])
        
    @h5_file_access  
    def link_file(self, fn, as_empty=True):
        """ 
        this might be useful when the buffer in the same holder does not work the best
        data from a different holder can be linked here
        all samples (top-level groups in the file) will be linked
        
        if as_empty is True, the external data are linked under the empty group
        """
        
        self.enable_write(True)

        with h5py.File(fn, "r") as fh5:
            # check for redundant sample names
            new_samples = list(fh5.keys())
            if as_empty:
                root = grp_empty_cells
                if not grp_empty_cells in list(self.fh5.keys()):
                    self.fh5.create_group(grp_empty_cells)
                    cur_samples = []
                else:
                    cur_samples = list(self.fh5[grp_empty_cells].keys())
            else:
                root = ''
                cur_samples = self.samples
            redundant_names = list(set(cur_samples) & set(new_samples))
            if len(redundant_names)>0:
                print("linking not allowed, found redundant sample: ", redundant_names)
            else:
                for sn in new_samples:
                    self.fh5[f'{root}/{sn}'] = h5py.ExternalLink(fn, sn) #SoftLink(fh5[sn])
    
        self.enable_write(False)     
    
    def assign_empty(self, empty_dict):
        """
        assign the empty cell scattering, the sn is user-specified, could come from the scan metadata
        consider saving that info during data collection
        
        for now, prepare the dictionary from the data collection spreadsheet, similar to buffer_list/sb_dict
        """        
        self.assign_sample_attr(empty_dict, "empty")
        
    @h5_file_access  
    def get_empty_d1(self, sn, input_grp="averaged"):
        ens = self.empty_list[sn][0]   # saved as a list, but there should be only one element
        d1s,attrs = get_d1s_from_grp(self.fh5[f'.empty/{ens}/processed'], self.qgrid, ens)
        d1b = d1s[input_grp]
        return d1b
    
    @h5_file_access  
    def get_empty_d0(self, sn, attr, debug=False):
        ens = self.empty_list[sn][0]   # saved as a list, but there should be only one element
        if debug:
            print(f'reading {attr} from .empty/{ens}/processed/attrs/', 
                  self.fh5[f'.empty/{ens}/processed/attrs'].keys())
        return self.fh5[f'.empty/{ens}/processed/attrs/{attr}'][...]
    
    @h5_file_access  
    def subtract_empty(self, samples, input_grp="averaged", max_distance=50, debug=False):
        """ this should be based on monitor counts strictly
            input_grp="merged": look at the data frame by frame, use self.d0s[sn]['selection'] (must exist)
            input_grp="averaged": take the average sample data and empty data
        """
        if samples is None:
            samples = list(self.empty_list.keys())
        elif isinstance(samples, str):
            if sn not in list(self.empty_list.keys()): 
                return
            samples = [samples]
        
        if self.read_only:
            print("h5 file is read-only ...")
            return
        
        if debug is True:
            print("start processing: subtract_empty()")
            t1 = time.time()

        for sn in samples:
            d1es = self.get_empty_d1(sn, input_grp)
            if input_grp=="averaged":
                d1c = self.d1s[sn][input_grp].bkg_cor(d1es,  debug=debug)
            else:
                if not 'selection' in self.d0s[sn].keys():
                    raise Exception(f"define selection for averaging {sn} first ...")
                elif len(self.d0s[sn]['selection'])!=len(d1es):
                    raise Exception(f"incorrect selection length for {sn}: {len(d1es)} vs {self.d0s[sn]['selection']}")
                
                d1ss = self.d1s[sn][input_grp]
                d1cs0 = [d1ss[i].bkg_cor(d1es[i]) for i in range(len(d1ss))]                               
                sel = filter_by_similarity(d1cs0, max_distance=max_distance, 
                                            preselection=self.d0s[sn]['selection'], debug=debug)
                d1cs = d1cs0[sel]
                if len(sel)==1:
                    d1c = d1cs[0]
                else:
                    d1c = d1cs[0].avg(d1cs[1:])
                
            d1c.label = f"{sn}-empty_subtracted"
            self.d1s[sn]['empty_subtracted'] = d1c

        self.save_d1s() 
        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
            
    def process(self, detectors=None, update_only=False,
                reft=-1, sc_factor=1., save_1d=False, save_merged=False, 
                filter_data=True, selection=None, debug=False, N = 8):
        """
        subtract empty first and populate d1s[sn]['empty_subtracted']
        """
        if filter_data=="keep":
            self.load_d1s()
        self.load_data(update_only=update_only, detectors=detectors, reft=reft, 
                       save_1d=save_1d, save_merged=save_merged, debug=debug, N=N)
        if filter_data=="keep":
            self.average_d1s(update_only=update_only, filter_data=False, selection=None, debug=debug)
        else:
            self.average_d1s(update_only=update_only, filter_data=filter_data, selection=selection, debug=debug)
        self.set_trans(trans_mode.external)
        self.subtract_empty(self.samples)
        self.subtract_buffer(update_only=update_only, sc_factor=sc_factor, input_grp='empty_subtracted', debug=debug)
        
    def plot_d1s(self, sn, show_subtracted='empty_subtracted', **kwargs):
        if show_subtracted=='empty_subtracted':
            src_d1 = self.d1s[sn]['averaged']
            bkg_d1 = self.get_empty_d1(sn)
        elif show_subtracted=='subtracted':
            src_d1 = self.d1s[sn]['empty_subtracted']
            bn = self.buffer_list[sn][0]  # this does not accomordate multiple buffers
            bkg_d1 = self.d1s[bn]['empty_subtracted']
        else:
            show_subtracted = False
            src_d1 = None
            bkg_d1 = None
        h5xs.plot_d1s(self, sn, show_subtracted=show_subtracted, src_d1=src_d1, bkg_d1=bkg_d1, **kwargs)
     
