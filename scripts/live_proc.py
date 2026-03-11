#!/nsls2/conda/envs/2024-3.0-py310-tiled/bin/python
"""
SAXSstrm/WAXSstrm (stream_data): 
    get data from SAXS/WAXS detectors through ZMQ and buffer in queue 
    queue should be under stream_data
    also saves raw data files??
    clear queue when saving file; one queue perfile???
processor (data_merger) 
    take data from the queue under the stream_data objects
    produce and buffer processed data
    process in chunks?? use multi-processing
plotter
    produce live plots
    take whatever is at the end of the processor queue
file save should be in chunks??? extensible size    
use Redis to save det_config, data_path, and other information
    qgrid_1d, q_bin_ranges_2d, phi_N
    maybe also recent UIDs and the corresponding sample info/header; so less to do when pack_h5()??

    update since the previous version:
        update q-grid from redis
        add processing muscle diffraction, plotting only 
    
"""

import sys
shared_dir = '/nsls2/software/mx/lix/pylibs'
#shared_dir = '/nsls2/users/lyang/pro/pylibs'
sys.path = [shared_dir]+sys.path

import zmq,json
from py4xs.hdf import h5exp
from py4xs.detector_config import create_det_from_attrs
from py4xs.data2d import Data2d,MatrixWithCoords,unflip_array
from py4xs.slnxs import Data1d
from py4xs.hdf import merge_d1s
from py4xs.utils import get_grid_from_bin_ranges
import multiprocessing
from threading import Thread
from collections import OrderedDict
import time,pathlib
from collections import deque
import matplotlib
matplotlib.use("Agg")
import pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
import fast_histogram as fh
import redis,os
import h5py
from ophyd import EpicsSignal
from scipy.signal import find_peaks

default_q_bin_ranges = [[[0.0045, 0.1105], 106], [[0.1125, 2.9025], 558]]
#qgrid = get_grid_from_bin_ranges(q_bin_ranges)
default_qgrid_1d = np.hstack((np.arange(0.005, 0.0499, 0.001),
                   np.arange(0.05, 0.0999, 0.002),
                   np.arange(0.1, 0.4999, 0.005),
                   np.arange(0.5, 0.9999, 0.01),
                   np.arange(1.0, 3.2, 0.03)))

qgrids = np.arange(-0.15, 0.1501, 0.001)
qgridw = np.arange(-1.1, 1.1001, 0.0075)
default_qxy_grids = [qgrids, qgridw]

default_phi_N = 32

#redis_host = "xf16id-ioc2"
redis_host = "epics-services-lix"
redis_port = 6379

MAX_PROCESSES = 50 # Max Number of processes to run
DEBUG = False

def proc_merge1d(imgs, parms, frn0, sn):
    """ utility function to perfrom azimuthal average and merge detectors
    """
    ndet,exps,sc,qgrid = parms
    ret = []

    nframes = len(imgs)
    for n in range(nframes):
        d1s = []
        for i in range(ndet):
            dt = Data1d()
            dt.load_from_2D(imgs[n][i], exps[i], qgrid, 
                            mask=exps[i].mask, save_ave=False, label=f"{sn}, frn#{n+frn0}")
            dt.scale(sc[i])
            d1s.append(dt)
    
        dm = merge_d1s(d1s, None)   # does not need detector info
        ret.append(dm)
            
    return ret

def proc_merge2d(imgs, parms):
    ndet,dQ,dPhi,dMask,dQPhiMask,dWeight,cQPhiMask,bin_ranges = parms
    q_bin_ranges,q_N,phi_range,phi_N = bin_ranges       
    nframes = len(imgs)
    ret = np.zeros(shape=(nframes, phi_N, q_N))
    
    for n in range(nframes):
        #print(imgs[n][0][0][0])  # for testing, this should show the frame number 
        for i in range(ndet):
            data = imgs[n][i]
            # there should be no overlap between SAXS and WAXS, not physically possible
            mm = np.vstack([fh.histogram2d(dQ[i], dPhi[i], bins=(qN, phi_N), range=[qrange, phi_range], 
                                           weights=data[dMask[i]]) for qrange,qN in q_bin_ranges]).T
            mm[dQPhiMask[i]]/=dWeight[i][dQPhiMask[i]]
            ret[n][mm>0] = mm[mm>0]
        ret[n][cQPhiMask] = np.nan

    return ret

def proc_merge(imgs, parms, frn0, sn, data_type_2d='qphi'):
    """ returns both 1D and 2D data
        qROI is for extracting phi profile
    """
    ndet,dQ,dPhi,dMask,dQPhiMask,dWeight,cQPhiMask,bin_ranges,qROI = parms
    q_bin_ranges,q_N,phi_range,phi_N = bin_ranges       
    nframes = len(imgs)
    ret2d = np.zeros(shape=(nframes, phi_N, q_N))     # q-phi maps
    ret1d_q = np.zeros(shape=(nframes, q_N))          # q profiles
    ret1d_phi = np.zeros(shape=(nframes, phi_N))      # phi profiles
    
    for n in range(nframes):
        #print(imgs[n][0][0][0])  # for testing, this should show the frame number 
        for i in range(ndet):
            data = imgs[n][i]
            # there should be no overlap between SAXS and WAXS, not physically possible
            mm = np.vstack([fh.histogram2d(dQ[i], dPhi[i], bins=(qN, phi_N), range=[qrange, phi_range], 
                                           weights=data[dMask[i]]) for qrange,qN in q_bin_ranges]).T
            mm[dQPhiMask[i]]/=dWeight[i][dQPhiMask[i]]
            ret2d[n][mm>0] = mm[mm>0]
        ret2d[n][cQPhiMask] = np.nan

    if nframes>1 and DEBUG:
        # save the raw/processed data into a temporary file 
        ts = time.asctime().split()[3].replace(":", "")
        with h5py.File(f"/nsls2/data/lix/legacy/softioc-data/tmp_data-{ts}.h5", "w") as fh5:
            #shp = imgs[0][0].shape
            #fh5.create_dataset(f"raw/d",
            #                   shape=(nframes,2,shp[0],shp[1]), chunks=True, compression='lzf')
            #fh5[f"raw/d"][...] = np.asarray(imgs)        
            fh5.create_dataset(f"processed/qphi/merged/d",
                               shape=(nframes,phi_N,q_N), chunks=True, compression='lzf')
            fh5[f"processed/qphi/merged/d"][...] = ret            
    
    return ret2d,ret1d_q,ret1d_phi

def proc_fiber(data, qxy_grids, detectors,
               qr_grid=np.linspace(0, 0.06, 101), align_peak_q_range=[0.03, 0.06]):
    """ data is a single set of SAXS/WAXS images
        qxy_grids is a list, grids with range/resolution appropriate for SAXS/WAXS data
        qr_grid, align_peak_q_range are for finding the euqator in the SAXS pattern
    """
    print("entering proc_fiber()")
    mqxys = []
    Phi0 = None
    
    for img,det in zip(data,detectors):
        d2 = Data2d(img, exp=det.exp_para)
        if Phi0 is None:
            d2.conv_Iqphi(Nq=qr_grid, Nphi=121, mask=d2.exp.mask)
            dd1 = d2.qphi_data.apply_symmetry()
            d0 = dd1.line_profile("y", xrange=align_peak_q_range, return_data1d=True) 
        
            idx0 = 5  # avoid finding a peak at the beginng of end of [-180, 180] span
            ret = find_peaks(d0.data[idx0:-idx0], distance=20, prominence=10)[0]
        
            if len(ret)>0: 
                idx1 = ret[0]+idx0  # this is an index
                #Phi0 = d0.qgrid[idx1]   # weighted average should be more accurate
                Phi0 = np.average(d0.qgrid[idx1-idx0:idx1+idx0], weights=d0.data[idx1-idx0:idx1+idx0])
            else:
                Phi0 = 0
    
        Qx = d2.exp.Q*np.cos(np.radians(d2.exp.Phi-Phi0))
        Qy = d2.exp.Q*np.sin(np.radians(d2.exp.Phi-Phi0))
        Qxi = d2.exp.Q*np.cos(np.radians(d2.exp.Phi-Phi0+180.))
        Qyi = d2.exp.Q*np.sin(np.radians(d2.exp.Phi-Phi0+180.))

        for qxy_grid in qxy_grids:
            mqxy0 = d2.data.conv(qxy_grid, qxy_grid, Qx, Qy, mask=d2.exp.mask)
            mqxy1 = d2.data.conv(qxy_grid, qxy_grid, Qxi, Qyi, mask=d2.exp.mask)        
            mqxy2 = mqxy0.merge([mqxy1])
            mqxy3 = mqxy2.copy()
            mqxy3.d = np.flipud(mqxy2.d)
            mqxy = mqxy2.merge([mqxy3])
            mqxy.d /= det.fix_scale
            mqxy.xc_label = "qx"
            mqxy.yc_label = "qy"
            mqxys.append(mqxy)

    # mqxy[0] is the high-res SAXS data
    qxy_SAXS = mqxys[0]
    # mqxy[1] is the low-res SAXS data
    # mqxy[3] is the WAXS data, mqxy[2] is not usable
    #qxy_WAXS = mqxy[3].merge([mqxy[1]])
    qxy_WAXS = mqxys[3]

    # meridional profile
    md1s = mqxys[0].line_profile('y', xrange=[-0.02, 0.02], yrange=[0, 0.2], return_data1d=True)
    md1w = mqxys[3].line_profile('y', xrange=[-0.1, 0.1], yrange=[0., 1.], return_data1d=True)
    qmw = np.min(md1w.qgrid[md1w.data>0])
    qms = np.max(md1s.qgrid[md1s.data>0])
    if qms>qmw:
        qms = qmw
    qgrid = np.hstack([md1s.qgrid[md1s.qgrid<qms], 
                       [qms, qmw], 
                       md1w.qgrid[md1w.qgrid>=qmw]])
    data = np.hstack([md1s.data[md1s.qgrid<qms], #/detectors[0].fix_scale, 
                      [np.nan, np.nan], 
                      md1w.data[md1w.qgrid>=qmw]]) #/detectors[1].fix_scale])
    idx = ((data>0) & (qgrid>0.005))
    d1_mer = Data1d()
    d1_mer.qgrid = qgrid[idx]
    d1_mer.data = data[idx]

    # equitorial
    ed1s = mqxys[0].line_profile('x', xrange=[0.007, 0.035], yrange=[-0.02, 0.02], return_data1d=True)
    idx = ((ed1s.data>0) & (ed1s.qgrid>0.005))
    d1_equ = Data1d()
    d1_equ.qgrid = ed1s.qgrid[idx] 
    d1_equ.data = ed1s.data[idx] #/detectors[0].fix_scale

    return qxy_SAXS,qxy_WAXS,d1_equ,d1_mer


class streaming_data:    
    def __init__(self, host, det_spec, stype=zmq.SUB):
        """ 
        host: currently tcp://xf16idc-pilatus1m:1234 and tcp://xf16idc-pilatus900k:1234
        det_spec: either SAXS or WAXS
        stype: zmq.SUB for PUB/SUB, all subscribers receive data
               zmq.PULL for PUSH/PULL, must be pushed to the next consumer
        """
        self.context = zmq.Context()
        self.sock = self.context.socket(stype)
        if stype == zmq.SUB:
            self.sock.setsockopt(zmq.SUBSCRIBE, b'')
        self.sock.connect(host)
        print('Client %s %s %s'%(('connect to', host, stype)))

        self.data_queue = deque()
        if det_spec=="SAXS":
            self.h5path_prefix="pil1M"
            self.h5fn_ext="_SAXS"
        elif det_spec=="WAXS":
            self.h5path_prefix="pilW2"
            self.h5fn_ext="_WAXS2"
        else:
            raise Exception(f"can't figure out how to deal with {det_spec} data from {host}")
        
        # start a thread
        #print(f"read to receive streaming data from {host}")
        Thread(target=self.ingest, daemon=True).start()
    
    def ingest(self):
        while True:
            header = self.sock.recv()
            if sys.hexversion >= 0x03000000:
                header = header.decode()

            #info = json.loads(header)
            info = json.loads(header.replace('\r\n', ''))   # '\r\n' in TIFF header cause problems
            # receive data
            data = np.frombuffer(self.sock.recv(), dtype=str(info['type'])).reshape(info['shape'])
            #print(info)
            frn = info['frame']
            ndattr = info['ndattr']
            #print(ndattr)
            # example of ndattr
            # {
            #  "DriverFileName":"/ramdisk/pil_pilW2/current_01709_WAXS2_00009.cbf",
            #  "CBF header":"uid=ad23d2ec-03c5-4e56-ac4b-5aea9d5d0e9b",
            #  "HDF path":"/ramdisk/hdf/pilW2/2024-3/315168/313955_3//Holder9_02/",
            #  "HDF filename":"sample343_WAXS2",
            #  "NumImages": 10}
            # }
            # the corresponding proc_path in this case should be /nsls2/data/lix/[proposals/legacy]/2024-3/pass-315168/313955_3
            uid = ndattr['CBF header'][4:]
            nimgs = ndattr['NumImages']

            # make data available
            #print(f"adding data to queue: {hdr}, {frn}")
            self.data_queue.append([uid, frn, nimgs, data])

class data_merger:
    
    fig_path = "/nsls2/data/lix/legacy/softioc-data/"
    save_data_type = {"scanning": "qphi",
                      "muscle": "qxqy"}
    
    def __init__(self):
        self.results_2d = None   # for plotting
        self.d1s_plot = deque()
        self.proc_config = ""
        
        self.data_cache = []     # for processing then saving

        self.detectors = []
        self.qgrid_config_ts = 0
        self.det_config_ts = 0

        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.hdrs = OrderedDict({})

        self.update_qgrid()
        
        self.processing_queue = deque()
        self.max_d1s_in_plot=400
        # self.processing_queue = multiprocessing.Queue()
        self.current_uid = ""
        self.processing_pool = multiprocessing.Pool(processes=MAX_PROCESSES)
        Thread(target=self.file_writer, daemon=True).start()
    
    def update_qgrid(self):
        print("updating qgrid_config ...")
        qgrid_attr = json.loads(self.redis.get('qgrid_config'))
        if qgrid_attr is not None:
            self.q_bin_ranges_2d = qgrid_attr['q_bins']
            self.qgrid_1d = get_grid_from_bin_ranges(self.q_bin_ranges_2d)
            self.phi_N = qgrid_attr['phi_N']            
        else:
            self.qgrid_1d = default_qgrid_1d
            self.q_bin_ranges_2d = default_q_bin_ranges_2d
            self.phi_N = default_phi_N
        
        print("setting qgrid/phi_N: ", self.q_bin_ranges_2d, self.phi_N)
        self.qgrid_2d = get_grid_from_bin_ranges(self.q_bin_ranges_2d)
        self.phi_range = [-180, 180]
        self.phigrid = np.linspace(-180, 180, self.phi_N)
    
    def file_writer(self):
        """ keep checking the processing_queue, save data when they become available
        """
        Nphi = self.phi_N
        Nq = len(self.qgrid_2d)
        
        while True:
            while len(self.processing_queue)==0:
                time.sleep(1)

            nimgs,uid,frn1,th = self.processing_queue.popleft()
            hdr = self.hdrs[uid]
            proc_path = hdr['proc_path'] 
            proc_dir = f"{proc_path}/processed"
            sn = hdr['sample_name']
            fn = f"{proc_dir}/{sn}-Iqphi.h5"

            if not os.path.exists(proc_dir):   
               os.makedirs(proc_dir)  # otherwise this thread will die when accessing fn
            
            #print(f"receiving data for {uid}, frn# {frn1} ... ") #, end="")
            output = th.get()
            nfrn = output.shape[0]
            print(f"received images {frn1-nfrn}-{frn1}, {output.shape} .")

            if os.path.isfile(fn) and frn1==nfrn:    # try not writing to exisitng file
                os.rename(fn, fn+"~")                # maybe should delete altogether
            with h5py.File(fn, "a") as fh5:
                try:
                    dset = fh5[f"{sn}/processed/qphi/merged/d"]
                except:  # new file
                    dets_attr = [det.pack_dict() for det in self.detectors]
                    fh5.attrs['detectors'] = json.dumps(dets_attr)
                    fh5.attrs['qgrid'] = list(self.qgrid_2d)
                    fh5.attrs['Nphi'] = Nphi
                    print("save dset shape:  ", (nimgs,Nphi,Nq))
                    print("parms2d:  ", self.parms2d[-1])
                    dset = fh5.create_dataset(f"{sn}/processed/qphi/merged/d", shape=(nimgs,Nphi,Nq),
                                              chunks=True, compression='lzf')
                    grp = fh5[f"{sn}/processed/qphi"]
                    grp.attrs['type'] = "MatrixWithCoords"
                    grp.attrs['xc_label'] = 'q'
                    grp.attrs['xc'] = self.qgrid_2d
                    grp.attrs['yc_label'] = 'phi'
                    grp.attrs['yc'] = self.phigrid
                    grp['merged'].attrs['len'] = nimgs
                
                print(f"saving array {output.shape} into dataset {dset.shape} in {fn}: ")
                print(f"    starting from frn#{frn1-nfrn}, mean={np.nanmean(output)}")
                dset[frn1-nfrn:frn1,:,:] = output

            del th
            
    def update_header(self):
        """ uid is for the current data
            also check proc_config here
        """
        pc = exp_config.get(as_string=True)
        if pc!=self.proc_config:
            self.proc_config = pc
            self.d1s_plot.clear()
            if pc=="muscle":
               self.max_d1s_in_plot = 50
            else:
               self.max_d1s_in_plot = 400  # for tracking SEC data or sinogram shape
        
        hdrs = self.redis.get('scan_info')
        if hdrs is not None:
            self.hdrs = json.loads(hdrs, object_pairs_hook=OrderedDict)
            uids = list(self.hdrs.keys())
            self.sn = self.hdrs[uids[-1]]['sample_name']
            
        ts = float(self.redis.get('qgrid_config_timestamp'))
        if ts>self.qgrid_config_ts:
            self.qgrid_config_ts = ts
            self.update_qgrid()
        ts = float(self.redis.get('det_config_timestamp'))
        if ts>self.det_config_ts or ts>self.qgrid_config_ts:
            self.det_config_ts = ts
            self.update_detectors()
    
    def update_detectors(self):
        print("read detector config from Redis...")

        dets_attr = self.redis.get('det_config')
        if dets_attr is not None:
            self.detectors = [create_det_from_attrs(attrs) for attrs in json.loads(dets_attr)] 
            
            dQ = {}
            dPhi = {}
            exps = {}
            dMask = {}
            dQPhiMask = {}
            dWeight = {}
            ndet = len(self.detectors)
            sc = {}
            
            for i in range(ndet):
                sc[i] = 1./self.detectors[i].fix_scale
                
                exp = self.detectors[i].exp_para
                exps[i] = exp
                dMask[i] = ~unflip_array(exp.mask.map, exp.flip)
                dQ[i] = unflip_array(exp.Q, exp.flip)[dMask[i]]
                dPhi[i] = unflip_array(exp.Phi, exp.flip)[dMask[i]]
                ones = np.ones_like(dQ[i])
                dWeight[i] = np.vstack([fh.histogram2d(dQ[i], dPhi[i], bins=(qN, self.phi_N), 
                                                       range=[qrange, self.phi_range], weights=ones) 
                                     for qrange,qN in self.q_bin_ranges_2d]).T
                dWeight[i] *= self.detectors[i].fix_scale
                dQPhiMask[i] = (dWeight[i]>0)
                
            cQPhiMask = ~dQPhiMask[0]
            for i in range(1,ndet):
                cQPhiMask &= ~dQPhiMask[i]       
                
            self.parms1d = [ndet,exps,sc,self.qgrid_1d]
            bin_ranges = [self.q_bin_ranges_2d,len(self.qgrid_2d),self.phi_range,self.phi_N] 
            self.parms2d = [ndet,dQ,dPhi,dMask,dQPhiMask,dWeight,cQPhiMask,bin_ranges]
        
    def save_fiber_plots(self, sc0=1.1, nd1s_full=40, d1s_roi=[0.05, 0.08], dpi=200):
        proc1d_tmp_path = pathlib.Path(f"{self.fig_path}/proc1d.tmp")
        proc1d_final_path = pathlib.Path(f"{self.fig_path}/proc1d.png")
        proc2d_tmp_path = pathlib.Path(f"{self.fig_path}/proc2d.tmp")
        proc2d_final_path = pathlib.Path(f"{self.fig_path}/proc2d.png")
        
        fig = plt.figure(figsize=(8,5))
        gs = gridspec.GridSpec(1, 3)
        ax1 = fig.add_subplot(gs[0:2])
        ax2 = fig.add_subplot(gs[2:])

        sc = 1.
        for d1_equ,d1_mer in list(self.d1s_plot)[-nd1s_full:]:
            sc /= sc0
            ax1.plot(d1_mer.qgrid, d1_mer.data*sc)
            ax2.plot(d1_equ.qgrid, d1_equ.data*sc)
        ax1.plot(d1_mer.qgrid, d1_mer.data*sc, "b-", linewidth=2)
        ax2.plot(d1_equ.qgrid, d1_equ.data*sc, "b-", linewidth=2)

        ax1.set_xscale('symlog', linthresh=0.18)
        ax1.set_yscale('log')
        ax1.set_yticklabels([])
        ax1.set_xticks(np.array([1,2,3,4,5,6,7,8,9,10])*2.*3.142/429, minor=True)
        ax1.xaxis.grid(True, which='minor')
        ax1.set_title('Meridonial')

        ax2.set_xticks(0.2*3.142/np.array([32,34,36,38,40,42,44]), minor=True)
        ax2.xaxis.grid(True, which='minor')
        ax2.set_yscale('log')
        ax2.set_yticklabels([])
        ax2.xaxis.set_ticks_position("bottom")
        ax2.yaxis.set_ticks_position("right")
        ax2.set_title('Equatorial')

        plt.savefig(proc1d_tmp_path, format="png", dpi=dpi)
        plt.close()
        proc1d_tmp_path.rename(proc1d_final_path)
        
        qxy_SAXS,qxy_WAXS = self.results_2d 
        fig,axs = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
        qxy_SAXS.plot(axs[0], norm="symlog", nolabel=True)
        qxy_WAXS.plot(axs[1], norm="symlog", nolabel=True)
        plt.suptitle(f'{self.sn}, frn# {self.last_frn}') #, y=0.92)
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.05, top=0.93, bottom=0.05)
        plt.savefig(proc2d_tmp_path, format="png", dpi=dpi)
        plt.close()
        proc2d_tmp_path.rename(proc2d_final_path)

    def save_plots(self, sc0=1.1, nd1s_full=40, d1s_roi=[0.05, 0.08]):
        proc1d_tmp_path = pathlib.Path(f"{self.fig_path}/proc1d.tmp")
        proc1d_final_path = pathlib.Path(f"{self.fig_path}/proc1d.png")
        proc2d_tmp_path = pathlib.Path(f"{self.fig_path}/proc2d.tmp")
        proc2d_final_path = pathlib.Path(f"{self.fig_path}/proc2d.png")
        
        #print(f"saving plots in {self.fig_path} ...")
        fig = plt.figure()
        gs = gridspec.GridSpec(4, 1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1:])

        sc = 1.
        for d1 in list(self.d1s_plot)[-nd1s_full:]:
            sc /= sc0
            ax2.loglog(d1.qgrid, d1.data*sc)
        #plt.ylim(bottom=0.8)
        ax2.plot(d1.qgrid, d1.data*sc, "b-", label=d1.label, linewidth=3)
        ax2.legend(frameon=False)

        idx = (d1.qgrid>d1s_roi[0])&(d1.qgrid<d1s_roi[1])
        ax1.plot([np.nansum(d1.data[idx]) for d1 in self.d1s_plot])
        ax1.xaxis.set_ticks_position("top")
        
        plt.savefig(proc1d_tmp_path, format="png")
        plt.close()
        proc1d_tmp_path.rename(proc1d_final_path)
        
        dm = MatrixWithCoords()
        dm.d = self.results_2d
        dm.xc = self.qgrid_2d
        dm.xc_label = "q" 
        dm.yc = self.phigrid
        dm.yc_label = "phi" 
        dm = dm.apply_symmetry()
        plt.figure()
        dm.plot(ax=plt.gca(), norm="symlog", sc_factor="x1.5")
        plt.title(f'{self.sn}, frn# {self.last_frn}')
        plt.savefig(proc2d_tmp_path, format="png")
        plt.close()
        proc2d_tmp_path.rename(proc2d_final_path)

    def proc_full_data(self, uid, nimgs, frn1, data, end_of_scan=False):
        """ don't want to start too many background processes
            each scan should not contain more than 10000 frames
            process every 500 frames??
            do 2D only for now
        """
        self.data_cache += data
        #print(f"place holder for processing in the background, cached {len(self.data_cache)} frames ...")
        if len(self.data_cache)>100 or end_of_scan:
            print(f"dispatching processing job ...")
            th = self.processing_pool.apply_async(proc_merge2d, (np.asarray(self.data_cache), self.parms2d))
            self.processing_queue.append([nimgs, uid, frn1, th])
            self.data_cache = []       
        
    def proc(self, uid, nimgs, pdata, end_of_scan=False):
        """ pdata is a list of (sfrn, [sdata, wdata])

            take only the first frame to generate live plot
            send the full data to be processed in the background
        """
        data = []
        for sfrn,[sdata, wdata] in pdata:
            data.append([sdata, wdata])
        self.last_frn = pdata[-1][0]
            
        frn0,[sdata, wdata] = pdata[0]
        frn1,[sdata, wdata] = pdata[-1]
        #print(f"in proc(), fr# {frn0} - {frn1}")
        
        if self.proc_config=="muscle":
            print("processing ...")
            qxy_SAXS,qxy_WAXS,d1_equ,d1_mer = proc_fiber(data[-1], default_qxy_grids, self.detectors)
            self.results_2d = [qxy_SAXS,qxy_WAXS]
            self.d1s_plot += deque([[d1_equ,d1_mer]])    
            while len(self.d1s_plot)>self.max_d1s_in_plot:
                self.d1s_plot.popleft()
            self.save_fiber_plots()            
        else:
            ret1d = proc_merge1d([data[-1]], self.parms1d, int(pdata[0][0]), self.sn)
            self.results_2d = proc_merge2d([data[-1]], self.parms2d)[0]        
            self.d1s_plot += deque(ret1d)    
            while len(self.d1s_plot)>self.max_d1s_in_plot:
                self.d1s_plot.popleft()
            self.save_plots()

        if nimgs==frn1:
            end_of_scan = True

        # skip this step for solution scattering
        if self.proc_config in ['scanning']:
            self.proc_full_data(uid, nimgs, frn1, data, end_of_scan)


SAXSstrm = streaming_data("tcp://xf16id-ioc2:9001" , "SAXS")
WAXSstrm = streaming_data("tcp://xf16id-ioc2:9002" , "WAXS")

processor = data_merger()
uid0 = ""

update_interval = 1   # update every two seconds
t0 = time.time()

exp_config = EpicsSignal("XF:16IDC-ES:EMconfig")
#processor.proc_config = exp_config.get(as_string=True)
processor.proc_config = 'unknown'

try:
    while True:
        nfrn = np.min([len(SAXSstrm.data_queue), len(WAXSstrm.data_queue)])
        t1 = time.time()    
        if t1-t0<=update_interval or nfrn==0:
            time.sleep(0.5)
        else:
            t0 = t1
            suid,sfrn,nimgs,sdata = SAXSstrm.data_queue[0]
            wuid,wfrn,nimgs,wdata = WAXSstrm.data_queue[0]
            print(time.asctime(), f"processing {suid}, frn# {sfrn}, {nfrn} frames")
            
            if suid!=wuid:
                # this should be rare, not worth the time to deal with it, restart the script between scans
                raise Exception(f"mismatched scan for SAXS/WAXS data: {suid} vs {wuid}")
            if sfrn!=wfrn: 
                print(f"mismatched frn for SAXS/WAXS data: {sfrn} vs {wfrn}, synchronizing ...  ", end="")
                while sfrn<wfrn:
                    while len(SAXSstrm.data_queue)==0:
                        time.sleep(0.5)
                    _ = SAXSstrm.data_queue.popleft()
                    suid,sfrn,nimgs,sdata = SAXSstrm.data_queue[0]
                while sfrn>wfrn:
                    while len(WAXSstrm.data_queue)==0:
                        time.sleep(0.5)
                    _ = WAXSstrm.data_queue.popleft()
                    wuid,wfrn,nimgs,wdata = WAXSstrm.data_queue[0]
                print('Done.')

            pdata = []
            for i in range(nfrn):
                suid,sfrn,nimgs,sdata = SAXSstrm.data_queue.popleft()
                wuid,wfrn,nimgs,wdata = WAXSstrm.data_queue.popleft()
                #sdata[0][0] = sfrn  # for testing, this should show the frame number # don't remember what this does, but it's causing trouble

                if suid!=uid0: # new scan
                    if len(pdata)>0:
                        processor.proc(suid, nimgs, pdata, end_of_scan=True)
                    # write data
                    #processor.save_data(c[uid].metadata['start']['sample_name'])
                    pdata = []
                    uid0 = suid
                    processor.update_header() 
                    
                pdata.append([sfrn, [sdata, wdata]])
            processor.proc(suid, nimgs, pdata)
            #if len(c[uid])>0: # run finished, write data
            #    processor.save_data(c[uid].metadata['start']['sample_name'])
except Exception as e:
    print(f"Exception in loop {e}")
finally:
    processor.processing_pool.close()
    processor.processing_pool.join()
