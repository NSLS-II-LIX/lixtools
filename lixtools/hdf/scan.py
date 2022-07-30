from py4xs.data2d import Data2d,MatrixWithCoords
from py4xs.slnxs import Data1d
from py4xs.hdf import lsh5,h5xs
from py4xs.utils import run
import lixtools
import numpy as np
import multiprocessing as mp
import json,os,copy,tempfile,re
import tomopy

from .an import h5xs_an

def calc_tomo(args):
    an,mm,kwargs = args
    algorithm = kwargs.pop("algorithm")
    proj = mm.d.reshape((len(mm.yc),1,len(mm.xc)))
    rot_center = tomopy.find_center(proj, np.radians(mm.yc))
    recon = tomopy.recon(proj, np.radians(mm.yc), center=rot_center, algorithm=algorithm, sinogram_order=False, **kwargs)
    #recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
    
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
            self.attrs[sn]['scan'] = json.loads(self.fh5[sn].attrs['scan'])
            
    def import_raw_data(self, fn_raw, sn=None, force_uniform_steps=True, prec=0.001, exp=1,
                        force_synch='auto', force_synch_trig=0, **kwargs):
        """ create new group, copy header, get scan parameters, calculate q-phi map
        """
        if not isinstance(fn_raw, list):
            fn_raw = [fn_raw]
        
        for fnr in fn_raw:
            sns = super().import_raw_data(fnr, save_attr=["source", "header", "scan"], 
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
                                 
    def make_map_from_attr(self, sname="overall", attr_names="transmission", 
                           ref_int_map="int_saxs", correct_for_transsmission=True, recalc_trans_map=True):
        """ for convenience in data processing, all attributes extracted from the data are saved as
            proc_data[sname]["attrs"][attr_name]
            
            for visiualization and for further data processing (e.g. run tomopy), these attributes need
            to be re-organized (de-snaking, merging) to reflect the shape of the scan/sample view
            
            sn="overall" is reserved for merging data from partial scans
            this seems not necessary to produce maps for individual files if there are more than one
            
            attr_names can be a string or a list
        """

        if sname=="overall":
            samples = self.samples
        elif sname in self.samples:
            samples = [sname]
        else:
            raise exception(f"sample {sname} does not exist") 
        
        if isinstance(attr_names, str):
            attr_names = [attr_names]
        
        # must have transmission data if correct_for_transsmission, or if need to calculate absorption
        if correct_for_transsmission or "absorption" in attr_names:
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
                    print(f"de-snaking {sn}, {an}      \r", end="")
                    for i in range(1,self.attrs[sn]['scan']['shape'][0],2):
                        m.d[i] = np.flip(m.d[i])
                if "maps" not in self.proc_data[sn].keys():
                    self.proc_data[sn]['maps'] = {}
                maps.append(m)
                self.proc_data[sn]['maps'][an] = m

            # assume the scans are of the same type, therefore start from the same direction
            if sname=="overall":
                mm = maps[0].merge(maps[1:])
                if "overall" not in self.proc_data.keys():
                    self.proc_data['overall'] = {}
                    self.proc_data['overall']['maps'] = {}
                self.proc_data['overall']['maps'][an] = mm
        
        if correct_for_transsmission:
            for an in attr_names:
                if an in ["transmission", "absorption"]:
                    continue
                print(f"transmmision correction: {sname}, {an}      \r", end="")
                self.proc_data[sname]['maps'][an].d /= self.proc_data[sname]['maps']["transmission"].d

        if 'absorption' in attr_names:
            if not ref_int_map in self.proc_data['overall']['maps'].keys():
                raise Exception(f"cannot find ref_int_map: {ref_int_map}")
            d = self.proc_data[sname]['maps'][ref_int_map].d
            h,b = np.histogram(d[~np.isnan(d)], bins=100)
            vbkg = (b[0]+b[1])/2
            mm = self.proc_data[sname]['maps']['transmission'].copy()
            t1 = np.average(mm.d[self.proc_data[sname]['maps'][ref_int_map].d<vbkg])
            mm.d = -np.log(mm.d/t1)
            mm.d[mm.d<0] = 0
            self.proc_data[sname]['maps']['absorption'] = mm
        
        print()
        self.save_data(save_sns=[sname], save_data_keys=["maps"])
        
    def calc_tomo_from_map(self, attr_names, debug=True, **kwargs):
        """ attr_names can be a string or a list
            ref_int_map is used to figure out where tranmission value should be 1
        """
        if len(self.h5xs)>1:
            sn = "overall"
        else:
            sn = samples[0]
        
        if isinstance(attr_names, str):
            attr_names = [attr_names]
            
        if not "tomo" in self.proc_data[sn]:
            self.proc_data[sn]['tomo'] = {}
            
        pool = mp.Pool(len(attr_names))
        jobs = []
        for an in attr_names:
            if debug:
                print(f"processing {sn}, {an}           \r", end="")
            mm = self.proc_data[sn]['maps'][an]
            tm = mm.copy()
            tm.yc = tm.xc
            tm.xc_label = "x"
            tm.yc_label = "y"
            self.proc_data[sn]['tomo'][an] = tm
            jobs.append(pool.map_async(calc_tomo, [(an, mm, kwargs)]))
        
        pool.close()
        for job in jobs:
            an,data = job.get()[0]
            self.proc_data[sn]['tomo'][an].d = data
            print(f"data received for {an}                \r", end="")
        pool.join()
            
        if debug:
            print(f"saving data                         ")        
        self.save_data(save_sns=sn, save_data_keys=["tomo"])

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
