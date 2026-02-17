#!/nsls2/software/mx/lix/conda_envs/an_2024Aug/bin/python

import sys,os
shared_dir = '/nsls2/software/mx/lix/pylibs'
#shared_dir = '/nsls2/users/lyang/pro/pylibs'
sys.path = [os.getcwd(), shared_dir]+sys.path

import numpy as np
import h5py,hdf5plugin
from py4xs.hdf import h5xs,h5exp,lsh5
from py4xs.detector_config import create_det_from_attrs
from lixtools.hdf import h5xs_an,h5xs_scan
from py4xs.utils import get_grid_from_bin_ranges
from lixtools.mapping.common import get_q_data_dask,sub_bkg_q_dask,get_phi_data_dask
from lixtools.mapping.common import prep_XRF_data,scf,fix_absorption_map,make_maps_from_Iq,scale_transmitted_maps
from lixtools.mapping.plants import calc_CI,prep_mfa_data
from lixtools.tomo.common import stack_2d_slices

import glob,PIL,time,json
import importlib
from tomopy.recon.rotation import find_center_vo

if len(sys.argv)==1:
    ppfn = "proc_param.py"
else:
    ppfn = sys.argv[1]
print("loading parameters from ", ppfn)
pp = importlib.import_module(ppfn.rstrip(".py"))

if not hasattr(pp, 'stacked_tomo_file'):
    print(f"stacked_tomo_file is not defined in {ppfn} ...")
    exit()  

coords = []
if hasattr(pp, 'ylist'):
    coors = pp.ylist

fns = [f"{sample}{pp.proc_ext}_an2.h5" for sample in pp.sample_list]

stack_2d_slices(fns, pp.stacked_tomo_file, coords=coords, replace_path=pp.replace_path)

