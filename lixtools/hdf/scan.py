from py4xs.data2d import Data2d,MatrixWithCoords
from py4xs.slnxs import Data1d
from py4xs.hdf import lsh5,h5xs
from py4xs.utils import run
import lixtools,h5py
import numpy as np
import multiprocessing as mp
import json,os,copy,tempfile,re
import tomopy

from .an import h5xs_an

from scipy.signal import find_peaks

def hop(phi, I, plot=False):
    # first need to find the peak and trim data range to -90 to 90 deg
    # make sure that that phi covers the full azimuthal angle range
    if np.max(phi)-np.min(phi)<360:
        raise Exception(f"phi range too narrow: {np.min(phi), np.max(phi)}")
    pks,_ = find_peaks(I, height=np.max(I)/2, distance=0.4*len(I))    
    for pk in pks:    
        if pk>len(phi)/4:
            break
    phi0 = phi[pk]
    
    idx = tuple([(phi>=phi0-90) & (phi<=phi0+90)])
    phi00 = np.sum(I[idx]*phi[idx])/np.sum(I[idx])
    
    # all angular position should fall within [-90, 90]
    phi1 = phi-phi00
    idx = (phi1<-90)
    idx1 = (phi1>90)
    phi1[idx] -= 180*(np.floor(phi1[idx]/180+0.5))
    phi1[idx1] -= 180*(np.floor(phi1[idx1]/180+0.5))

    if plot:
        plt.figure()
        plt.plot(phi1,I,'.')
    
    phi1 = np.radians(phi1)
    c2b = np.sum(I*np.cos(phi1)**2)/(np.sum(I))
    return (2.*c2b-1)/2

def get_hop_from_map(d, xrange, plot=False):
    phi,I,ee = d.apply_symmetry().line_profile(direction="y", xrange=xrange)
    return hop(phi, I, plot)

def get_roi(d, qphirange):
    return np.nanmean(d.apply_symmetry().roi(*qphirange).d)


def calc_tomo(args):
    an,mm,kwargs = args
    
    # filter out incomplete data
    idx = ~np.isnan(mm.d.sum(axis=1))
    dmap = mm.d[idx, :]
    yc = mm.yc[idx]
    xc = mm.xc
    proj = dmap.reshape((len(yc),1,len(xc)))
    
    if "algorithm" in kwargs.keys():
        algorithm = kwargs.pop("algorithm")
    else:
        algorithm = "gridrec"
        
    if "center" in kwargs.keys():
        rot_center = kwargs.pop("center")
    else:
        rot_center = tomopy.find_center(proj, np.radians(yc))
    
    recon = tomopy.recon(proj, np.radians(yc), center=rot_center, algorithm=algorithm, sinogram_order=False, **kwargs)
    recon = tomopy.circ_mask(recon, axis=0) #, ratio=0.95)
    
    return [an,recon[0,:,:]]

class h5xs_scan(h5xs_an):
    """ keep the detector information
        import data from raw h5 files, keep track of the file location
        copy the meta data, convert raw data into q-phi maps
        can still show data, 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                           ref_int_map=None, ref_trans = 0.213,
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
            maps = []
            
            for sn in self.h5xs.keys():
                if not 'scan' in self.attrs[sn].keys():
                    get_scan_parms(self.h5xs[sn], sn)
                if an not in self.proc_data[sn]['attrs'].keys():
                    raise Exception(f"attribue {an} cannot be found for {sn}.")
                
                data = self.proc_data[sn]['attrs'][an].reshape(self.attrs[sn]['scan']['shape'])
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
                self.add_proc_data(sn, map_data_key, an, m)

            # assume the scans are of the same type, therefore start from the same direction
            if save_overall:
                mm = maps[0].merge(maps[1:])
                #if "overall" not in self.proc_data.keys():
                #    self.proc_data['overall'] = {}
                #    self.proc_data['overall'][map_data_key] = {}
                #if not map_data_key in self.proc_data['overall'].keys():
                #    self.proc_data['overall'][map_data_key] = {}
                #self.proc_data['overall'][map_data_key][an] = mm
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
                elif ref_trans>0: 
                    mm = self.proc_data[sname][map_data_key]['transmission'].copy()
                    mm.d = -np.log(mm.d/ref_trans)
                    mm.d[mm.d<0] = 0
                    mm.d[np.isnan(mm.d)] = 0
                    self.proc_data[sname][map_data_key]['absorption'] = mm
                else:
                    raise Exception("Don't know how to calculate absorption.")
        
        if debug: print()
        self.save_data(save_data_keys=[map_data_key], quiet=(not debug))
        
    def calc_tomo_from_map(self, attr_names, map_data_key='maps', tomo_data_key='tomo', debug=True, **kwargs):
        """ attr_names can be a string or a list
            ref_int_map is used to figure out where tranmission value should be 1
        """
        if len(self.h5xs)>1:
            sn = "overall"
        else:
            sn = self.samples[0]
        
        if isinstance(attr_names, str):
            attr_names = [attr_names]
            
        if not tomo_data_key in self.proc_data[sn]:
            self.proc_data[sn][tomo_data_key] = {}
            
        pool = mp.Pool(len(attr_names))
        jobs = []
        for an in attr_names:
            if debug:
                print(f"processing {sn}, {an}           \r", end="")
            mm = self.proc_data[sn][map_data_key][an]
            tm = mm.copy()
            tm.yc = tm.xc
            tm.xc_label = "x"
            tm.yc_label = "y"
            self.proc_data[sn][tomo_data_key][an] = tm
            jobs.append(pool.map_async(calc_tomo, [(an, mm, kwargs)]))
        
        pool.close()
        for job in jobs:
            an,data = job.get()[0]
            md = self.proc_data[sn][map_data_key][an].d
            data *= np.sum(md[np.isfinite(md)])/np.sum(data[np.isfinite(data)])
            self.proc_data[sn][tomo_data_key][an].d = data
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
