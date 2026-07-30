#!/nsls2/software/mx/lix/conda_envs/tf/bin/python
#
# this requires GPU
# XRF tomographic reconstruction with self-absorption correction
# assuming 4-channel detector 
#

import sys,os
#shared_dir = '/nsls2/software/mx/lix/pylibs'
shared_dir = '/nsls2/users/lyang/pro/pylibs'
sys.path = [os.getcwd(), shared_dir]+sys.path

import numpy as np
import h5py,hdf5plugin
from py4xs.hdf import h5xs,h5exp,lsh5
from lixtools.hdf import h5xs_an,h5xs_scan
from lixtools.tomo.xrfcor import XRFmlem

import glob,PIL,time,json
import importlib
from tomopy.recon.rotation import find_center_vo

if len(sys.argv)==1:
    ppfn = "proc_param.py"
else:
    ppfn = sys.argv[1]
print("loading parameters from ", ppfn)
pp = importlib.import_module(ppfn.rstrip(".py"))

def check_h5an2_validity(fn):
    if not os.path.exists(fn):
        return False
 
    with h5py.File(fn, "r") as fh:
        val = set(fh.attrs).issuperset(['Nphi', 'detectors', 'pre_proc', 'qgrid'])

    return val

for sample in pp.sample_list:
    print(f"processing {sample} ...")

    fn0 = f"{sample}{pp.proc_ext}_an2.h5"
    fns = sorted(glob.glob(f"{sample}{pp.rawdata_file_pattern}.h5"))
    sns = [fn[:-3] for fn in fns]

    # first check if a valid _an2 file exists
    if check_h5an2_validity(fn0):
        dt = h5xs_scan(fn0, replace_path=pp.replace_path)
    else:
        raise Exception(f"this script can only operate on valid ")
        
    sns = dt.samples
    if pp.save_overall:
        sn = 'overall'
        sns = [sn]
    else:
        sn = sns[0]

    #if not maps_present:
    dt.load_data("overall", ['maps'])
    print(dt.proc_data['overall']['maps'].keys())
    if pp.rot_cen is None:
        try:
            rot_cen = dt.get_h5_attr("overall/maps", "rot_cen")
            print("found recorded rotation center: ", rot_cen)
        except:
            rot_cen = find_center_vo(dt.proc_data['overall']['maps'][pp.ref_map].d)
            print("rotation center from tomopy: ", rot_cen)
    else:
        rot_cen = pp.rot_cen
    dt.set_h5_attr("overall/maps", "rot_cen", rot_cen)

    print(f"3c. reconstructing XRF tomograms with absorption correction ...")
    XRFmlem = XRFmlem(dt)
    xrf_list = []
    for ele in pp.ele_list: 
        print(f"processing {ele} ...")
        mlem.calc_abs(ele)
        mm1 = mlem.dt.proc_data['overall']['tomo']['absorption'].copy()
        en = ele.split('_')[0]
        mm1.d = mlem.recon(f"xrf_{en}", n_iter=pp.num_iter).numpy()
        k = f"xrf_{ele}_cor"
        xrf_list += k
        mlem.dt.add_proc_data("overall", "tomo", k, mm1)        

    dt.save_data("overall", ["tomo"], xrf_list)