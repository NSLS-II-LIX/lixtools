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
from lixtools.mapping.common import get_q_data_dask,sub_bkg_q_dask,get_phi_data_dask,make_map_from_overall_attr
from lixtools.mapping.common import prep_XRF_data,scf,fix_absorption_map,make_maps_from_Iq,scale_transmitted_maps,fix_maps
from lixtools.mapping.plants import calc_CI,prep_mfa_data
from lixtools.tomo.common import stack_2d_slices

import glob,PIL,time,json
import importlib
from tomopy.recon.rotation import find_center_vo

import warnings
warnings.filterwarnings("ignore")

if len(sys.argv)==1:
    ppfn = "proc_param.py"
else:
    ppfn = sys.argv[1]
print("loading parameters from ", ppfn)
pp = importlib.import_module(ppfn.rstrip(".py"))

de = h5exp("exp.h5")
q_bin_ranges = [[[0.0045, 0.1105], 106], [[0.1125, 2.9025], 558]]
default_qgrid = get_grid_from_bin_ranges(q_bin_ranges)

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

    # get detector config
    if pp.look_for_qphi_maps:
        #detectors = de.detectors
        #qgrid = default_qgrid
        #Nphi = 61
        with h5py.File(f"processed/{sns[0]}-Iqphi.h5") as fh5:
            dets_attr = fh5.attrs['detectors']
            qgrid = fh5.attrs['qgrid']
            Nphi = fh5.attrs['Nphi']
            detectors = [create_det_from_attrs(attrs) for attrs in json.loads(dets_attr)]
    else:
        detectors = de.detectors
        qgrid = default_qgrid
        Nphi = 61

    # first check if a valid _an2 file exists
    if check_h5an2_validity(fn0):
        dt = h5xs_scan(fn0, replace_path=pp.replace_path)
    else:
        dt = h5xs_scan(fn0, [detectors, qgrid], 
                       Nphi=Nphi, pre_proc="2D") 
    # the _an2 file should contain links to individual scans
    if not dt.has_proc_data(sns[-1], 'attrs', 'transmission'):
        if not dt.has_data('attrs', 'transmission') or len(dt.samples)<len(sns):
            for i in range(len(fns)):
                dt.import_raw_data(fns[i], sn=sns[i], force_uniform_steps=True, prec=0.001, debug=True)
            dt.save_data()
        else:
            dt.load_data(read_data_keys=['attrs'])

    if not dt.has_data("qphi"):
        pp.make_qphi_maps = True
    
    if pp.make_qphi_maps:
        print(f"0a. create q-phi maps ...")
    
        t0 = time.time()
        if pp.look_for_qphi_maps:
            # live-processed I-q maps are saved under porcessed, simply make links
            pfns = [f"processed/{sn}-Iqphi.h5" for sn in sns]
            #try:
            dt.link_proc_data(pfns)
            #except:
            #     print("sometime this happens even if the link is created already ...")
            #     print("pretend nothing is wrong for now ...")
        else:
            dt.process(N=16, max_c_size=2048, debug=True)
            print(f"time elapsed: {time.time()-t0:.1f} sec")
        dt.explicit_close_h5()

    sn0 = dt.samples[0]
    if pp.save_overall:
        sn = 'overall'
    else:
        sn = sn0
    if pp.extract_q_profiles:
        print(f"1a. extract q- profiles ...")
        get_q_data_dask(dt, q0=0.06, ex=2.3, q_range=pp.qrange_1d, dezinger=pp.dz, save_to_overall=pp.save_overall)
        d1dk = "merged"
        if pp.subtract_bkg_q:
            sub_bkg_q_dask(dt, save_to_overall=pp.save_overall)
            d1dk = "subtracted"
        dt.load_data(sn, ['Iq'], [d1dk])
        dt.proc_data[sn]['Iq']['averaged'] = np.average(dt.proc_data['overall']['Iq'][d1dk], axis=0)
        dt.save_data(save_sns=[sn], save_data_keys=["Iq"], save_sub_keys=['averaged'])
    
    if pp.extract_phi_profiles:
        print(f"1b. extract phi- profiles ...")
        for phik in pp.phi_profile_locations.keys():
            get_phi_data_dask(dt, q_range=pp.phi_profile_locations[phik], sub_key=phik) 

    maps_present = False
    if pp.construct_scattering_maps:
        print(f"2a. create absorption maps ...")
        dt.load_data(read_data_keys=['attrs'])
        dt.make_map_from_attr(attr_names=["transmitted", "absorption", "incident"], 
                              correct_for_transsmission=False, save_overall=pp.save_overall)
        fix_absorption_map(dt)

        print(f"2b. create scattering maps ...")
        okeys = []
        for k in pp.attr_dict.keys():
            # e.g. {"hop": [False, get_hop_from_map, ["qphi", "merged"], {"xrange": [0.04, 0.06]}],
            #       "hop2": [True, hop, ["Iphi", "merged"], {}] }
            ops,func,args,kwargs = pp.attr_dict[k]
            if ops: 
                if not dt.has_proc_data(sn, args[0], args[1]):
                    dt.load_data(sn, [args[0]], [args[1]])
                attr = func(dt.proc_data[sn][args[0]][args[1]], **kwargs)
                make_map_from_overall_attr(dt, attr, sname=sn)
            else:
                okeys.append(k)
                for ssn in dt.samples:
                    if not dt.has_proc_data(ssn, args[0], args[1]):
                        print("loading", ssn, args)
                        dt.load_data(ssn, [args[0]], [args[1]])
                    dt.extract_attr(ssn, k, func, *args, **kwargs)
        dt.make_map_from_attr(save_overall=pp.save_overall, attr_names=okeys, correct_for_transsmission=False)
            
        if not pp.extract_q_profiles:
            dt.load_data(read_data_keys=['Iq'])
            if "subtracted" in dt.proc_data[sn]['Iq'].keys():
                d1dk = "subtracted"
            else:
                d1dk = "merged"
        make_maps_from_Iq([dt], int_data=d1dk, q_list=pp.q_map_dict, abs_cor=False, save_overall=pp.save_overall)
        maps_present = True

    if pp.construct_XRF_maps:
        print(f"2c. create XRF maps ...")
        prep_XRF_data(dt, pp.ele_list, pyxrf_param_fn=pp.pyxrf_param_fn, eNstart=pp.eNstart, eNend=pp.eNend, save_overall=pp.save_overall)
        maps_present = True

    if pp.tomo_reconstruction:
        if not maps_present:
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

        map_list = list(set(dt.proc_data['overall']['maps'].keys()) - set(['incident', 'transmission', 'transmitted']))    
        xrf_maps = [mn for mn in map_list if 'xrf' in mn]
        map_list = list(set(map_list) - set(xrf_maps+['absorption']))
        Iphi_maps = [mn for mn in map_list if ('Iphi' in mn or 'mfa' in mn)]
        xs_maps = list(set(map_list) - set(Iphi_maps))
        xrf_maps = [k for k in xrf_maps if not "_sum" in k]

        if 'absorption' in dt.proc_data['overall']['maps'].keys():
            print(f"3a. reconstructing absorption tomograms ...")
            dt.calc_tomo_from_map(attr_names=['absorption'], algorithm=pp.algorithm, num_iter=pp.num_iter, center=rot_cen)

        if len(xs_maps)>0 and pp.construct_scattering_tomo:
            print(f"3b. reconstructing scattering tomograms ...")
            dt.calc_tomo_from_map(attr_names=xs_maps, algorithm=pp.algorithm, num_iter=pp.num_iter, center=rot_cen)    

        if hasattr(pp, 'calc_CI'):
            if not set(pp.calc_CI).issubset(set(pp.q_map_dict)):
                print(f"don't know how to get data from {pp.calc_CI} for calc_CI")
            else:
                calc_CI(dt, cell_key=pp.calc_CI[0], amor_key=pp.calc_CI[1], ref_key=pp.ref_map, ref_cutoff=pp.ref_cutoff)    

        if pp.construct_Iphi_tomo:
            print(f"3d. reconstructing Iphi tomograms ...")
            dt.load_data('overall', ['Iphi'])
            for k in pp.phi_profile_locations.keys():
                prep_mfa_data(dt, rot_cen, algorithm=pp.algorithm, data_sub_key=k, 
                              num_iter=pp.num_iter, ref_key=pp.ref_map, ref_cutoff=pp.ref_cutoff)
        
        if pp.construct_xrf_tomo and len(xrf_maps)>0:
            print(f"3c. reconstructing XRF tomograms ...")
            if pp.sum_xrf_channels:
                for k in xrf_maps:
                    mm = dt.proc_data['overall']['maps'][k][0].copy()
                    mm.d = np.nansum([m0.d for m0 in dt.proc_data['overall']['maps'][k]], axis=0)
                    mm.d /= (dt.proc_data['overall']['maps']['incident'].d/pp.ref_incident)
                    dt.proc_data['overall']['maps'][k+"_sum"] = mm
                fix_maps(dt, map_sub_keys=[k+"_sum" for k in xrf_maps], fix_inf=True)    # sometimes em1/em2 reports 0; this also saves maps  
                dt.calc_tomo_from_map(attr_names=[k+"_sum" for k in xrf_maps], algorithm=pp.algorithm, 
                                      num_iter=pp.num_iter, center=rot_cen)
            else:
                dt.calc_tomo_from_map(attr_names=xs_maps, algorithm=pp.algorithm, 
                                      num_iter=pp.num_iter, center=rot_cen)
