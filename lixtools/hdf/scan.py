from py4xs.data2d import Data2d,MatrixWithCoords
from py4xs.slnxs import Data1d
from py4xs.hdf import lsh5,h5xs
import numpy as np
import multiprocessing as mp
import json,os,copy

from .an import h5xs_an

class h5xs_scan(h5xs_an):
    """ keep the detector information
        import data from raw h5 files, keep track of the file location
        copy the meta data, convert raw data into q-phi maps
        can still show data, 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for sn in self.samples:
            self.attrs[sn]['scan'] = json.loads(self.fh5[sn].attrs['scan'])
            
    def import_raw_data(self, fn_raw, sn=None, force_uniform_steps=True, prec=0.001,
                        force_synch=0, force_synch_trig=0, **kwargs):
        """ create new group, copy header, get scan parameters, calculate q-phi map
        """
        if not isinstance(fn_raw, list):
            fn_raw = [fn_raw]
        
        for fnr in fn_raw:
            sn = super().import_raw_data(fnr, save_attr=["source", "header", "scan"], 
                                         force_uniform_steps=force_uniform_steps, prec=prec, **kwargs)
            fast_axis = self.attrs[self.samples[0]]['scan']['fast_axis']['motor']
            exp = self.attrs[self.samples[0]]['header']['pilatus']['exposure_time']
            self.get_mon(sn=sn, trigger=fast_axis, exp=exp, 
                         force_synch=force_synch, force_synch_trig=force_synch_trig)
    
                                 
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
