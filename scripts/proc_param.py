import numpy as np

# samples to process
sn = "Bo2"
ylist = np.arange(22.35, 22.55, 0.01) 

sample_list = [f"{sn}_y{y0:.3f}" for y0 in ylist]

# processing steps
make_qphi_maps = False
extract_q_profiles = False            
extract_phi_profiles = True         
construct_scattering_maps = True    
construct_XRF_maps = False
tomo_reconstruction = True

# processing parameters
# additional identifier in the _an2 filename
proc_ext = ""
rawdata_file_pattern = "-??"
replace_path = {"legacy": "proposals"}

# 0. 2D q-phi intensity maps 
save_overall = True          # assemble from multiple scans 
look_for_qphi_maps = True    # use live-processed data

# 1a. line profiles, in q and phi
qrange_1d = [0.005, 2.6]
subtract_bkg_q = True
dz = None      # dezinger algorithm, options are '1d', 'min'

## q values to extract phi profiles, given as a dictionary
phi_profile_locations = {"int_waxs1": [1.37, 1.41], "int_waxs2": [1.85, 1.87]}

# 2a. absorption maps

# 2b. generate scattering-based intensity maps/sinograms
## based on intensities at specific q values
q_map_dict = {"int_saxs": [0.1, 0.11], "int_waxs1": [1.37, 1.41], "int_waxs2": [1.85, 1.87]}

## additional maps to be constructed using extrat_attr()
## values for each key are whether operate on overall data, the function to be used, and the args and kwargs
## e.g. attr_dict = {"int_saxs": [False, get_roi, ["qphi", "merged"], {qphirange: [0.02, 0.05, -180, 180]}}
attr_dict = {}

# 2c. XRF maps
ele_list = ['K_K', 'Ca_K', 'Mn_K', 'Fe_K', 'Cu_K', 'Zn_K']
pyxrf_param_fn = None
ref_incident = 1000
# energy range, x10 eV
eNstart=310
eNend=1800

# 3. tomographic reconstruction
algorithm = "mlem"   # "sirt"
num_iter = 10
rot_cen = None
ref_map = "int_saxs"
ref_cutoff = 0.002

# 3a. absorption


# 3b. scattering 
construct_scattering_tomo = True

# 3c. XRF 
construct_xrf_tomo = False
sum_xrf_channels = True

# 3d. Iphi
construct_Iphi_tomo = True

stacked_tomo_file = f"{sn}-3d.h5"
