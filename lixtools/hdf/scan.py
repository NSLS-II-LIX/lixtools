from py4xs.data2d import Data2d,MatrixWithCoords
from py4xs.slnxs import Data1d
from py4xs.hdf import lsh5,h5xs
from py4xs.utils import run
import lixtools,h5py
import numpy as np
import multiprocessing as mp
import json,os,copy,tempfile,re
from scipy.signal import find_peaks

from .an import h5xs_an
from lixtools.tomo.common import calc_tomo

class h5xs_scan(h5xs_an):
    """ keep the detector information
        import data from raw h5 files, keep track of the file location
        copy the meta data, convert raw data into q-phi maps
        can still show data, 
    """
    def __init__(self, fn, *args, **kwargs):
        super().__init__(fn, *args, **kwargs)

        for sn in self.samples:
            self.attrs[sn]['scan'] = json.loads(self.get_h5_attr(sn, 'scan'))
            
    def import_raw_data(self, fn_raw, sn=None, force_uniform_steps=True, prec=0.001, exp=1,
                        force_synch='auto', force_synch_trig={'em1': 'auto', 'em2': 'auto'}, **kwargs):
        """ create new group, copy header, get scan parameters, calculate q-phi map
        """
        if not isinstance(fn_raw, list):
            fn_raw = [fn_raw]
        
        for fnr in fn_raw:
            sns = super().import_raw_data(fnr, sn=sn, save_attr=["source", "header", "scan"], 
                                          force_uniform_steps=force_uniform_steps, prec=prec, **kwargs)
            fast_axis = self.attrs[self.samples[0]]['scan']['fast_axis']['motor']
            try:
                t = self.attrs[self.samples[0]]['header']['pilatus']['exposure_time']
                exp = t
            except:
                print(f"Unable to find meta data on exposure time, assuming {exp:.2f}s")
            for sn in sns:
                self.get_mon(sn=sn, trigger=fast_axis, exp=exp, 
                             force_synch=force_synch, force_synch_trig=force_synch_trig)    
                
    def get_index_from_map_coords(self, sn, xc, yc, map_name=None, map_data_key='maps'):
        """ sn of "overall" should be dealt with separately
        """
        if sn=="overall":
            raise Exception("not yet implemented for overall")
        elif not sn in self.samples:
            raise Exception(f"invalid sample name: {sn}")
        if map_name is None: # all maps under the map_data_key should share the same coordinates
            map_name = list(self.proc_data[sn][map_data_key].keys())[0]
            
        mm = self.proc_data[sn][map_data_key][map_name]
        im = mm.copy()
        im.d = np.arange(len(mm.xc)*len(mm.yc)).reshape(mm.d.shape)    
        if self.attrs[sn]['scan']['snaking']:
            for i in range(1,self.attrs[sn]['scan']['shape'][0],2):
                im.d[i] = np.flip(im.d[i])
        
        xb = (im.xc[1:]+im.xc[:-1])/2
        yb = (im.yc[1:]+im.yc[:-1])/2
        if xc>xb[-1]:
            ix = -1
        else:
            ix = np.where(xb>=xc)[0][0]
        if yc>yb[-1]:
            iy = -1
        else:
            iy = np.where(yb>=yc)[0][0]
        
        return im.d[iy,ix]
        
                
    def get_attr_from_map(self, sn, map_name, map_data_key='maps', quiet=True):
        """ it is sometimes necessary to translate a 2D map into a 1D array, so that the index of the array
            can match the index of the 2D scattering data
        """
        m = np.copy(self.proc_data[sn][map_data_key][map_name].d)
        if self.attrs[sn]['scan']['snaking']:
            if not quiet:
                print(f"re-snaking {sn}, {map_name}      \r", end="")
            for i in range(1,self.attrs[sn]['scan']['shape'][0],2):
                m.d[i] = np.flip(m.d[i])
                
        return m.flatten()
                                 
    def make_map_from_overall_attr(self, attr, correct_for_transsmission=True):
        """ this is used to calculate maps from attrubutes that are assigned to "overall" by simple stacking, and 
            therefore need to be reorganized to recover the actual shape of the data
            e.g. coefficients from NMF, or data extracted from the I(q) or I(phi) 1D profiles
            
            related to this, also need to translate between filename/frame number and coordinates in the maps
        """
        sl = 0
        maps = []
        for sn in dt.h5xs.keys():
            if not 'scan' in self.attrs[sn].keys():
                get_scan_parms(self.h5xs[sn], sn)

            ll = np.prod(self.attrs[sn]['scan']['shape'])
            m = MatrixWithCoords()
            #m.d = np.copy(self.proc_data['overall']['attrs'][an][sl:sl+ll].reshape(self.attrs[sn]['scan']['shape']))
            m.d = np.copy(attr[sl:sl+ll].reshape(self.attrs[sn]['scan']['shape']))
            sl += ll

            m.xc = self.attrs[sn]['scan']['fast_axis']['pos']
            m.yc = self.attrs[sn]['scan']['slow_axis']['pos']
            m.xc_label = self.attrs[sn]['scan']['fast_axis']['motor']
            m.yc_label = self.attrs[sn]['scan']['slow_axis']['motor']
            if self.attrs[sn]['scan']['snaking']:
                for i in range(1,self.attrs[sn]['scan']['shape'][0],2):
                    m.d[i] = np.flip(m.d[i])

            maps.append(m)

        # assume the scans are of the same type, therefore start from the same direction
        mm = maps[0].merge(maps[1:])
        #if "overall" not in self.proc_data.keys():
        #    self.proc_data['overall'] = {}
        #    self.proc_data['overall']['maps'] = {}
        #self.proc_data['overall']['maps'][an] = mm
        return mm
    
    def make_map_from_attr(self, save_overall=True, map_data_key='maps', attr_names="transmission", 
                           ref_int_map=None, ref_trans=-1, pk_prominence_cutoff = 500,
                           correct_for_transsmission=True, recalc_trans_map=True,
                           debug=True):
        """ for convenience in data processing, all attributes extracted from the data are saved as
            proc_data[sname]["attrs"][attr_name]
            
            for visiualization and for further data processing (e.g. run tomopy), these attributes need
            to be re-organized (de-snaking, merging) to reflect the shape of the scan/sample view
            
            sn="overall" is reserved for merging data from partial scans
            this seems not necessary to produce maps for individual files if there are more than one
            
            attr_names can be a string or a list
            
            ref_int_map can be used as a reference for zero absorption (e.g. SAXS intensity)
            alternatively, the known value of empty beam trasmission can be provided: ref_trans
            this value can be obtained by doing a histogram on the transmission values
        """

        if isinstance(attr_names, str):
            attr_names = [attr_names]
        
        # must have transmission data if correct_for_transsmission, or if need to calculate absorption
        if correct_for_transsmission or "absorption" in attr_names:
            sname = self.samples[0]
            if not "transmission" in attr_names:
                if not 'overall' in self.proc_data.keys():
                    attr_names.append("transmission")
                elif not "transmission" in self.proc_data[sname].keys():
                    attr_names.append("transmission")
                
        for an in attr_names:
            if an=="absorption":
                continue   # this should be calculated from absorption map
            if an=="transmission" and not recalc_trans_map:
                continue   # this should be calculated from absorption map
            
            for sn in self.h5xs.keys():
                if not 'scan' in self.attrs[sn].keys():
                    get_scan_parms(self.h5xs[sn], sn)
                if an not in self.proc_data[sn]['attrs'].keys():
                    raise Exception(f"attribue {an} cannot be found for {sn}.")
                
                # attr could be 2D, e.g. for multi-element XRF data
                attr_data = self.proc_data[sn]['attrs'][an]
                if len(attr_data.shape)==1:
                    attr_dim = 1
                    attr_data = [attr_data]
                else:
                    attr_dim = attr_data.shape[-1]
                    attr_data = [attr_data[:,i] for i in range(attr_dim)]
                maps = []
                for d0 in attr_data:
                    shp = self.attrs[sn]['scan']['shape']
                    data = d0[:np.prod(shp)].reshape(shp)   # sometimes the electrometers record the last circular buffer twice
                    m = MatrixWithCoords()
                    m.d = np.copy(data)
                    m.xc = self.attrs[sn]['scan']['fast_axis']['pos']
                    m.yc = self.attrs[sn]['scan']['slow_axis']['pos']
                    m.xc_label = self.attrs[sn]['scan']['fast_axis']['motor']
                    m.yc_label = self.attrs[sn]['scan']['slow_axis']['motor']
                    if self.attrs[sn]['scan']['snaking']:
                        #print(f"de-snaking {sn}, {an}      \r", end="")
                        for i in range(1,self.attrs[sn]['scan']['shape'][0],2):
                            m.d[i] = np.flip(m.d[i])
                    #if map_data_key not in self.proc_data[sn].keys():
                    #    self.proc_data[sn][map_data_key] = {}
                    maps.append(m)
                if len(maps)==1:
                    maps = maps[0]
                self.add_proc_data(sn, map_data_key, an, maps)

            # assume the scans are of the same type, therefore start from the same direction
            if save_overall:           
                if attr_dim>1:   # multiple maps per attribute
                    mm = []                
                    for i in range(attr_dim):
                        maps = [self.proc_data[sn][map_data_key][an][i] for sn in self.h5xs.keys()]
                        mm.append(maps[0].merge(maps[1:]))
                else:
                    maps = [self.proc_data[sn][map_data_key][an] for sn in self.h5xs.keys()]
                    mm = maps[0].merge(maps[1:])
                self.add_proc_data('overall', map_data_key, an, mm)
        
        if save_overall:
            sns = ['overall']
        else:
            sns = self.samples
        
        for sname in sns:
            # this should correct for transmitted intensity rather than for transmission
            # sometimes the beam can be off for part of the scan, that part of the data should not be corrected
            if correct_for_transsmission:  
                trans = np.copy(self.proc_data[sname][map_data_key]["transmitted"].d)
                idx = (trans<np.average(trans)/4)
                trans[idx] /= np.average(~idx)
                trans[~idx] = 1
                for an in attr_names:
                    if an in ["incident", "transmission", "absorption", "transmitted"]:
                        continue
                    if debug:
                        print(f"transmmision correction: {sname}, {an}      \r", end="")
                    #self.proc_data[sname][map_data_key][an].d /= self.proc_data[sname][map_data_key]["transmission"].d
                    self.proc_data[sname][map_data_key][an].d /= trans

            if 'absorption' in attr_names:
                if ref_int_map in self.proc_data[sname][map_data_key].keys():
                    d = self.proc_data[sname][map_data_key][ref_int_map].d
                    h,b = np.histogram(d[~np.isnan(d)], bins=100)
                    vbkg = (b[0]+b[1])/2
                    mm = self.proc_data[sname][map_data_key]['transmission'].copy()
                    t1 = np.average(mm.d[self.proc_data[sname][map_data_key][ref_int_map].d<vbkg])
                    mm.d = -np.log(mm.d/t1)
                    mm.d[mm.d<0] = 0
                    self.proc_data[sname][map_data_key]['absorption'] = mm
                else:
                    mm = self.proc_data[sname][map_data_key]['transmission'].copy()
                    if ref_trans<=0: 
                        # try to figure out the ref_trans value
                        dd = mm.d[np.isfinite(mm.d)].flatten() # mm.d[~np.isinf(mm.d)].flatten()
                        avg = np.average(dd)
                        std = np.std(dd)
                        hh,bb = np.histogram(dd, range=(avg-4*std, avg+4*std), bins=int(np.sqrt(len(dd))))
                        ##ref_trans = (bb[1:]+bb[:-1])[np.argmax(hh)]/2
                        peaks, properties = find_peaks(hh, width=2)
                        pk = -1
                        for i in reversed(range(len(peaks))):
                            if properties['prominences'][i]>pk_prominence_cutoff:
                                pk = peaks[i]
                                break
                        if pk<0:
                            print("could not automatically find ref_trans value ...")
                            ref_trans = -ref_trans
                        else:
                            ref_trans = (bb[pk]+bb[pk+1])/2
                    
                    mm.d = -np.log(mm.d/ref_trans)
                    mm.d[mm.d<0] = 0
                    mm.d[np.isinf(mm.d)] = np.nan
                    self.proc_data[sname][map_data_key]['absorption'] = mm
                #else:
                #    raise Exception("Don't know how to calculate absorption.")
        
        if debug: print()
        self.save_data(save_data_keys=[map_data_key], quiet=(not debug))
        
    def calc_tomo_from_map(self, attr_names, map_data_key='maps', tomo_data_key='tomo', debug=True, **kwargs):
        """ attr_names can be a string or a list
            ref_int_map is used to figure out where tranmission value should be 1
        """
        if 'overall' in self.proc_data.keys():
            sn = "overall"
        else:
            sn = self.samples[0]
        
        if isinstance(attr_names, str):
            attr_names = [attr_names]
            
        if not tomo_data_key in self.proc_data[sn].keys():
            self.proc_data[sn][tomo_data_key] = {}
            
        pool = mp.Pool(len(attr_names))
        jobs = []
        for an in attr_names:
            if debug:
                print(f"processing {sn}, {an}           \r", end="")
            mm = self.proc_data[sn][map_data_key][an]
            if isinstance(mm, list):
                mlen = len(mm)
                tm = mm[0].copy()
            else:
                mlen = 1
                tm = mm.copy()
            tm.yc = tm.xc
            tm.xc_label = "x"
            tm.yc_label = "y"
            
            if mlen==1:
                self.proc_data[sn][tomo_data_key][an] = tm
                jobs.append(pool.map_async(calc_tomo, [(an, mm, kwargs)]))
            else:
                self.proc_data[sn][tomo_data_key][an] = []
                for i in range(mlen):
                    self.proc_data[sn][tomo_data_key][an].append(tm.copy())
                    jobs.append(pool.map_async(calc_tomo, [(f"{an}_{i}", mm[i], kwargs)]))
        
        pool.close()
        for job in jobs:
            an,data = job.get()[0]
            if mlen==1:
                md = self.proc_data[sn][map_data_key][an].d
                data *= np.sum(md[np.isfinite(md)])/np.sum(data[np.isfinite(data)])
                self.proc_data[sn][tomo_data_key][an].d = data
            else:
                an,i = an.rsplit('_',1)
                i = int(i)
                md = self.proc_data[sn][map_data_key][an][i].d
                data *= np.sum(md[np.isfinite(md)])/np.sum(data[np.isfinite(data)])
                self.proc_data[sn][tomo_data_key][an][i].d = data
            if debug:
                print(f"data received for {an}                \r", end="")
        pool.join()
            
        if debug:
            print(f"saving data                         ")        
        self.save_data(save_sns=sn, save_data_keys=[tomo_data_key], quiet=(not debug))

def gen_scan_report(fn, client=None):
    dn = os.path.dirname(fn)
    if dn == "":
        dn = "."

    bn = os.path.basename(fn).split('.')
    if bn[-1]!='h5':
        raise Exception(f"{bn} does not appear to be a h5 file.")
    bn = bn[0]

    pn = lixtools.__path__
    if isinstance(pn, list):
        pn = pn[0]

    tmp_dir = tempfile.gettempdir()
    fn0 = os.path.join(pn, f"hdf/template_scan_report.ipynb")
    fn1 = os.path.join(dn, f"{bn}_report.ipynb")

    print("preparing the notebook ...")
    ret = run(["cp", fn0, fn1])
    # sed only works with unix
    #ret = run(["sed", "-i", f"s/00template00/{bn}/g", fn1])
    with open(fn1, 'r+') as fh:
        txt = fh.read()
        txt = re.sub('00template00.h5', fn, txt)
        if client:
            txt = re.sub('00scheduler_addr00', client.scheduler.addr, txt)
        fh.seek(0)
        fh.write(txt)
        fh.truncate()

    fn2 = os.path.join(tmp_dir, f"{bn}_report.html")
    print("executing ...")
    ret = run(["jupyter", "nbconvert", 
               fn1, f"--output={fn2}", 
               "--ExecutePreprocessor.enabled=True", 
               "--TemplateExporter.exclude_input=True", "--to", "html"],
              debug=True)
    print("cleaning up ...")
    ret = run(["mv", fn2, dn])
    ret = run(["rm", fn1])
