from py4xs.hdf import h5xs,lsh5,h5_file_access
from py4xs.slnxs import trans_mode,estimate_scaling_factor
import numpy as np
import pylab as plt
import json,time,copy

from scipy.linalg import svd
from scipy.interpolate import splrep,sproot,splev,UnivariateSpline
from scipy.ndimage.filters import gaussian_filter
from sklearn.decomposition import NMF

def qgrid_labels(qgrid):
    dq = qgrid[1]-qgrid[0]
    gpindex = [0]
    gpvalues = [qgrid[0]]
    gplabels = []

    for i in range(1,len(qgrid)-1):
        dq1 = qgrid[i+1]-qgrid[i]
        if np.fabs(dq1-dq)/dq>0.01:
            dq = dq1
            gpindex.append(i)
            prec = int(-np.log(dq)/np.log(10))+1
            gpvalues.append(qgrid[i])
    gpindex.append(len(qgrid)-1)
    gpvalues.append(qgrid[-1])

    for v in gpvalues:
        prec = int(-np.log(v)/np.log(10))+2
        gplabels.append(f"{v:.{prec}f}".rstrip('0'))
    
    return gpindex,gpvalues,gplabels
 

class h5sol_HPLC(h5xs):
    """ single sample (not required, but may behave unexpectedly when there are multiple samples), 
        many frames; frames can be added gradually (not tested)
    """ 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dbuf = None
        self.updating = False   # this is set to True when add_data() is active
        
    @h5_file_access  
    def process_sample_name(self, sn, debug=False):
        #fh5 = h5py.File(self.fn, "r+")
        fh5 = self.fh5
        self.samples = lsh5(fh5, top_only=True, silent=(debug is not True))
        if sn==None:
            sn = self.samples[0]
        elif sn not in self.samples:
            raise Exception(sn, "not in the sample list.")
        
        return sn 
        
    def load_d1s(self):
        super().load_d1s(self.samples[0])
        # might need to update meta data??
        
    def normalize_int(self, ref_trans=-1):
        """ 
        """
        sn = self.samples[0]        
        if 'merged' not in self.d1s[sn].keys():
            raise Exception(f"{sn}: merged data must exist before normalizing intensity.")

        max_trans = np.max([d1.trans for d1 in self.d1s[sn]['merged']])
        if max_trans<=0:
            raise Exception(f"{sn}: run set_trans() first, or the beam may be off during data collection.")
        if ref_trans<0:
            ref_trans=max_trans
            
        for d1 in self.d1s[sn]['merged']:
            d1.scale(ref_trans/d1.trans)
        
    def process(self, update_only=False, ext_trans=False,
                reft=-1, save_merged=False, 
                filter_data=False, debug=False, N=8, max_c_size=0):
        """ load data from 2D images, merge, then set transmitted beam intensity
        """

        self.load_data(update_only=update_only, reft=reft, 
                       save_merged=save_merged, debug=debug, N=N, max_c_size=max_c_size)
        
        # This should account for any beam intensity fluctuation during the HPLC run. While
        # typically for solution scattering the water peak intensity is relied upon for normalization,
        # it could be problematic if sample scattering has features at high q.  
        if ext_trans and self.transField is not None:
            self.set_trans(transMode=trans_mode.external)
        else:
            self.set_trans(transMode=trans_mode.from_waxs) 
        self.normalize_int()

    def subtract_buffer_NMF(self, excluded_frames_list, sn=None, sc_factor=0.995,
                            gaussian_filter_width=None,
                            Nc=5, poly_order=2, smoothing_factor=0.04, fit_with_polynomial=False,
                            plot_fit=False, ax1=None, ax2=None, debug=False, 
                            init='nndsvd', **kwargs):
        """ similar to SVD subtraction, accept that NMF is used instead
            kwargs are for NMF
        """
        sn = self.process_sample_name(sn, debug=debug)
        if debug is True:
            print("start processing: subtract_buffer()")
            t1 = time.time()

        if isinstance(poly_order, int):
            poly_order = poly_order*np.ones(Nc, dtype=int)
        elif isinstance(poly_order, list):
            if len(poly_order)!=Nc:
                raise Exception(f"the length of poly_order ({poly_order}) must match Nc ({Nc}).")
        else:
            raise Exception(f"invalid poly_order: {poly_order}")

        if isinstance(smoothing_factor, float) or isinstance(smoothing_factor, int):
            smoothing_factor = smoothing_factor*np.ones(Nc, dtype=float)
        elif isinstance(smoothing_factor, list):
            if len(smoothing_factor)!=Nc:
                raise Exception(f"the length of smoothing_factor ({smoothing_factor}) must match Nc ({Nc}).")
        else:
            raise Exception(f"invalid smoothing_factor: {smoothing_factor}")
                    
        nf = len(self.d1s[sn]['merged'])
        all_frns = list(range(nf))
        ex_frns = []
        for r in excluded_frames_list.split(','):
            if r=="":
                break
            r1,r2 = np.fromstring(r, dtype=int, sep='-')
            ex_frns += list(range(r1,r2))
        bkg_frns = list(set(all_frns)-set(ex_frns))
        
        dd2s = np.vstack([d1.data for d1 in self.d1s[sn]['merged']]).T
        if gaussian_filter_width is not None:
            dd2s = gaussian_filter(dd2s, sigma=gaussian_filter_width)
        dd2b = np.vstack([dd2s[:,i] for i in bkg_frns]).T
        
        model = NMF(n_components=Nc, init=init, **kwargs) 
        W = model.fit_transform(dd2b)
        H = model.components_

        H0 = []
        for i,d in enumerate(H):
            if fit_with_polynomial:
                pf = np.poly1d(np.polyfit(bkg_frns, d, poly_order[i])) 
            else:
                pf = UnivariateSpline(bkg_frns, d, s=smoothing_factor[i]) #30
            H0.append(pf(all_frns))
        dd2c = W@np.vstack(H0)

        if plot_fit:
            if ax1 is None:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
            for i in reversed(range(Nc)):
                ax1.plot(bkg_frns, H[i], '.') #s[i]*U[:,i])
            for i in reversed(range(Nc)):
                ax1.plot(all_frns, H0[i]) #s[i]*U[:,i])  
            ax1.set_xlim(0, nf)
            ax1.set_xlabel("frame #")
            if ax2 is not None:
                for i in reversed(range(Nc)):
                    ax2.semilogy(self.qgrid, W.T[i]) 
                ax2.set_xlabel("q")                

        if self.read_only:
            print("h5 file is read-only ...")
        else:
            grp = f""
            self.attrs[sn]['sc_factor'] = sc_factor
            self.attrs[sn]['subtraction'] = 'method: nmf , '
            self.attrs[sn]['subtraction'] += f"excluded frames: {excluded_frames_list} , "
            if fit_with_polynomial:
                self.attrs[sn]['subtraction'] += f"poly_fit with poly_order={poly_order}"
            else:
                self.attrs[sn]['subtraction'] += f"spline with smoothing={smoothing_factor}"
        
        if 'subtracted' in self.d1s[sn].keys():
            del self.d1s[sn]['subtracted']
        self.d1s[sn]['subtracted'] = []
        
        # the protein volume fraction should not be constant across the peak
        d0=np.sum((dd2s-dd2c)[10:20,:], axis=0)
        scm = np.repeat([1.-(1.-sc_factor)*d0/np.max(d0)], dd2s.shape[0], axis=0)
        dd2s -= dd2c*scm
        for i in range(nf):
            d1c = copy.deepcopy(self.d1s[sn]['merged'][i])
            d1c.data = dd2s[:,i]
            self.d1s[sn]['subtracted'].append(d1c)
            
        self.save_d1s(sn, debug=debug)


    def subtract_buffer_SVD(self, excluded_frames_list, sn=None, sc_factor=0.995,
                            gaussian_filter_width=None,
                            Nc=5, poly_order=8, smoothing_factor=0.04, fit_with_polynomial=False,
                            plot_fit=False, ax1=None, ax2=None, debug=False):
        """ perform SVD background subtraction, use Nc eigenvalues 
            poly_order: order of polynomial fit to each eigenvalue
            gaussian_filter width: sigma value for the filter, e.g. 1, or (0.5, 3)
            if fit_with_polynomial is True, use polinomial fitting with poly_order 
                otherwise, use UnivariateSpline with smoothing_factor 
        """
        sn = self.process_sample_name(sn, debug=debug)
        if debug is True:
            print("start processing: subtract_buffer()")
            t1 = time.time()

        if isinstance(poly_order, int):
            poly_order = poly_order*np.ones(Nc, dtype=int)
        elif isinstance(poly_order, list):
            if len(poly_order)!=Nc:
                raise Exception(f"the length of poly_order ({poly_order}) must match Nc ({Nc}).")
        else:
            raise Exception(f"invalid poly_order: {poly_order}")

        if isinstance(smoothing_factor, float) or isinstance(smoothing_factor, int):
            smoothing_factor = smoothing_factor*np.ones(Nc, dtype=float)
        elif isinstance(poly_order, list):
            if len(smoothing_factor)!=Nc:
                raise Exception(f"the length of smoothing_factor ({smoothing_factor}) must match Nc ({Nc}).")
        else:
            raise Exception(f"invalid smoothing_factor: {smoothing_factor}")
                    
        nf = len(self.d1s[sn]['merged'])
        all_frns = list(range(nf))
        ex_frns = []
        for r in excluded_frames_list.split(','):
            if r=="":
                break
            r1,r2 = np.fromstring(r, dtype=int, sep='-')
            ex_frns += list(range(r1,r2))
        bkg_frns = list(set(all_frns)-set(ex_frns))
        
        dd2s = np.vstack([d1.data for d1 in self.d1s[sn]['merged']]).T
        if gaussian_filter_width is not None:
            dd2s = gaussian_filter(dd2s, sigma=gaussian_filter_width)
        dd2b = np.vstack([dd2s[:,i] for i in bkg_frns]).T
        
        U, s, Vh = svd(dd2b.T, full_matrices=False)
        s[Nc:] = 0

        Uf = []
        # the time-dependence of the eigen values are fitted to fill the gap (excluded frames) 
        # polynomial fits will likely produce unrealistic fluctuations
        # cubic (default, k=3) spline fits with smoothing factor provides better control
        #     smoothing factor: # of knots are added to reduce fitting error below the s factor??
        for i in range(Nc):
            if fit_with_polynomial:
                Uf.append(np.poly1d(np.polyfit(bkg_frns, U[:,i], poly_order[i])))
            else:
                Uf.append(UnivariateSpline(bkg_frns, U[:,i], s=smoothing_factor[i]))
        Ub = np.vstack([f(all_frns) for f in Uf]).T
        dd2c = np.dot(np.dot(Ub, np.diag(s[:Nc])), Vh[:Nc,:]).T

        if plot_fit:
            if ax1 is None:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
            for i in reversed(range(Nc)):
                ax1.plot(bkg_frns, np.sqrt(s[i])*U[:,i], '.') #s[i]*U[:,i])
            for i in reversed(range(Nc)):
                ax1.plot(all_frns, np.sqrt(s[i])*Uf[i](all_frns)) #s[i]*U[:,i])  
            ax1.set_xlim(0, nf)
            ax1.set_xlabel("frame #")
            if ax2 is not None:
                for i in reversed(range(Nc)):
                    ax2.plot(self.qgrid, np.sqrt(s[i])*Vh[i]) #s[i]*U[:,i])
                ax2.set_xlabel("q")                

        if self.read_only:
            print("h5 file is read-only ...")
        else:
            grp = f""
            self.attrs[sn]['sc_factor'] = sc_factor
            self.attrs[sn]['svd excluded frames'] = excluded_frames_list
            self.attrs[sn]['svd parameter Nc'] = Nc
            self.attrs[sn]['svd parameter poly_order'] = poly_order
        
        if 'subtracted' in self.d1s[sn].keys():
            del self.d1s[sn]['subtracted']
        self.d1s[sn]['subtracted'] = []
        dd2s -= dd2c*sc_factor
        for i in range(nf):
            d1c = copy.deepcopy(self.d1s[sn]['merged'][i])
            d1c.data = dd2s[:,i]
            self.d1s[sn]['subtracted'].append(d1c)
            
        self.save_d1s(sn, debug=debug)

        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))

    def subtract_buffer(self, buffer_frame_range, sample_frame_range=None, first_frame=0, 
                        sn=None, update_only=False, 
                        sc_factor=1., show_eb=False, debug=False):
        """ buffer_frame_range should be a list of frame numbers, could be range(frame_s, frame_e)
            if sample_frame_range is None: subtract all dataset; otherwise subtract and test-plot
            update_only is not used currently
            first_frame:    duplicate data in the first few frames subtracted data from first_frame
                            this is useful when the beam is not on for the first few frames
        """

        sn = self.process_sample_name(sn, debug=debug)
        if debug is True:
            print("start processing: subtract_buffer()")
            t1 = time.time()

        if type(buffer_frame_range) is str:
            f1,f2 = buffer_frame_range.split('-')
            buffer_frame_range = range(int(f1), int(f2))
            
        listb  = [self.d1s[sn]['merged'][i] for i in buffer_frame_range]
        listbfn = buffer_frame_range
        if len(listb)>1:
            d1b = listb[0].avg(listb[1:], debug=debug)
        else:
            d1b = copy.deepcopy(listb[0])           
            
        if sample_frame_range==None:
            # perform subtraction on all data and save listbfn, d1b
            if self.read_only:
                print("h5 file is read-only ...")
            else:
                self.attrs[sn]['buffer frames'] = listbfn
                self.attrs[sn]['sc_factor'] = sc_factor
            self.d1s[sn]['buf average'] = d1b
            if 'subtracted' in self.d1s[sn].keys():
                del self.d1s[sn]['subtracted']
            self.d1s[sn]['subtracted'] = []
            for d1 in self.d1s[sn]['merged']:
                d1t = d1.bkg_cor(d1b, plot_data=False, debug=debug, sc_factor=sc_factor)
                self.d1s[sn]['subtracted'].append(d1t) 
            if first_frame>0:
                for i in range(first_frame):
                    self.d1s[sn]['subtracted'][i].data = self.d1s[sn]['subtracted'][first_frame].data
            self.save_d1s(sn, debug=debug)   # save only subtracted data???
        else:
            lists  = [self.d1s[sn]['merged'][i] for i in sample_frame_range]
            if len(listb)>1:
                d1s = lists[0].avg(lists[1:], debug=debug)
            else:
                d1s = copy.deepcopy(lists[0])
            sample_sub = d1s.bkg_cor(d1b, plot_data=True, debug=debug, sc_factor=sc_factor, show_eb=show_eb)
            return sample_sub
        
        #if update_only and 'subtracted' in list(self.d1s[sn].keys()): continue
        #if sn not in list(self.buffer_list.keys()): continue

        if debug is True:
            t2 = time.time()
            print("done, time lapsed: %.2f sec" % (t2-t1))
            
    @h5_file_access  
    def get_chromatogram(self, sn, q_ranges=[[0.02,0.05]], flowrate=0, plot_merged=False,
                 calc_Rg=False, thresh=2.5, qs=0.01, qe=0.04, fix_qe=True):
        """ returns data to be plotted in the chromatogram
        """
        if 'subtracted' in self.d1s[sn].keys() and plot_merged==False:
            dkey = 'subtracted'
        elif 'merged' in self.d1s[sn].keys():
            if plot_merged==False:
                print("subtracted data not available. plotting merged data instead ...")
            dkey = 'merged'
        else:
            raise Exception("processed data not present.")
            
        data = self.d1s[sn][dkey]
        nd = len(data)
        #qgrid = data[0].qgrid
        
        exp_t = self.exp_time(sn)
        ts = self.get_ts(sn, exp_t)
        idx = [(self.qgrid>i_minq) & (self.qgrid<i_maxq) for [i_minq,i_maxq] in q_ranges]
        nq = len(idx)
        
        d_t = np.zeros(nd)
        d_i = np.zeros((nq, nd))
        d_rg = np.zeros(nd)
        d_s = []
        
        for i in range(len(data)):
            ti = (ts[i]-ts[0])/60
            #if flowrate>0:
            #    ti*=flowrate
            d_t[i] = ti
                
            for j in range(nq): 
                d_i[j][i] = data[i].data[idx[j]].sum()
            ii = np.max([d_i[j][i] for j in range(nq)])
            d_s.append(data[i].data)

            if ii>thresh and calc_Rg and dkey=='subtracted':
                i0,rg,_ = data[i].plot_Guinier(qs, qe, fix_qe=fix_qe, no_plot=True)
                d_rg[i] = rg
    
        # read HPLC data directly from HDF5
        d_hplc = {}
        if 'hplc' in list(self.fh5[sn].keys()):
            hplc_grp = self.fh5[sn+"/hplc/data"]
            fields = lsh5(self.fh5[sn+'/hplc/data'], top_only=True, silent=True)
            for fd in fields:
                d_hplc[fd] = self.fh5[sn+'/hplc/data/'+fd][...].T   # self.fh5[sn+'/hplc/data/'+fd].value.T
    
        return dkey,d_t,d_i,d_hplc,d_rg,np.vstack(d_s).T
    
            
    def plot_data(self, sn=None, 
                  q_ranges=[[0.02,0.05]], logROI=False, markers=['bo', 'mo', 'co', 'yo'],
                  flowrate=-1, plot_merged=False,
                  ymin=-1, ymax=-1, offset=0, uv_xscale=1, uv_yscale=1, showFWHM=False, 
                  calc_Rg=False, thresh=2.5, qs=0.01, qe=0.04, fix_qe=True,
                  plot2d=True, norm='symlog', clim=[1.e-3, 10.],
                  show_hplc_data=[True, False],
                  export_txt=False, debug=False, 
                  fig_w=8, fig_h1=2, fig_h2=3.5, fig=None, ax1=None, ax2=None):
        """ plot "merged" if no "subtracted" present
            q_ranges: a list of [q_min, q_max], within which integrated intensity is calculated 
            export_txt: export the scattering-intensity-based chromatogram
            
        """
        
        if ax1 is None:
            if plot2d:
                if fig is None:
                    fig = plt.figure(figsize=(fig_w, fig_h1+fig_h2))
                hfrac = 0.82                
                ht2 = fig_h1/(fig_h1+fig_h2)
                box1 = [0.1, ht2+0.05, hfrac, (0.95-ht2)*hfrac] # left, bottom, width, height
                box2 = [0.1, 0.02, hfrac, ht2*hfrac]
                ax1 = fig.add_axes(box1)
            else:
                plt.figure(figsize=(fig_w, fig_h2))
                ax1 = plt.gca()
        ax1a = ax1.twiny()
        ax1b = ax1.twinx()
        
        sn = self.process_sample_name(sn, debug=debug)
        if flowrate<0:  # get it from metadata
            md = self.md_dict(sn, md_keys=['HPLC'])
            if "HPLC" in md.keys():
                flowrate = float(md["HPLC"]["Flow Rate (ml_min)"])
            else: 
                flowrate = 0.5
        dkey,d_t,d_i,d_hplc,d_rg,d_s = self.get_chromatogram(sn, q_ranges=q_ranges, 
                                                             flowrate=flowrate, plot_merged=plot_merged, 
                                                             calc_Rg=calc_Rg, thresh=thresh, 
                                                             qs=qs, qe=qe, fix_qe=fix_qe)
        data = self.d1s[sn][dkey]
        nq = len(q_ranges)
        
        if ymin == -1:
            ymin = np.min([np.min(d_i[j]) for j in range(nq)])
        if ymax ==-1:
            ymax = np.max([np.max(d_i[j]) for j in range(nq)])
        if logROI:
            pl_ymax = 1.5*ymax
            pl_ymin = 0.8*np.max([ymin, 1e-2])
        else:
            pl_ymax = ymax+0.05*(ymax-ymin)
            pl_ymin = ymin-0.05*(ymax-ymin)

        if export_txt:
            # export the scattering-intensity-based chromatogram
            for j in range(nq):
                np.savetxt(f'{sn}.chrome_{j}', np.vstack((d_t, d_i[j])).T, "%12.3f")
            
        for j in range(nq):
            ax1.plot(d_i[j], 'w-')
        ax1.set_xlabel("frame #")
        ax1.set_xlim((0,len(d_i[0])))
        ax1.set_ylim(pl_ymin, pl_ymax)
        ax1.set_ylabel("intensity")
        if logROI:
            ax1.set_yscale('log')

        i = 0 
        for k,dc in d_hplc.items():
            if show_hplc_data[i]:
                ax1a.plot(np.asarray(dc[0])*uv_xscale+offset,
                         ymin+dc[1]/np.max(dc[1])*(ymax-ymin)*uv_yscale, label=k)
            i += 1
            #ax1a.set_ylim(0, np.max(dc[0][2]))

        if flowrate>0:
            ax1a.set_xlabel("volume (mL)")
        else:
            ax1a.set_xlabel("time (minutes)")
        for j in range(nq):
            ax1a.plot(d_t, d_i[j], markers[j], markersize=5, label=f'x-ray ROI #{j+1}')
        ax1a.set_xlim((d_t[0],d_t[-1]))
        leg = ax1a.legend(loc='upper left', fontsize=9, frameon=False)

        if showFWHM and nq==1:
            half_max=(np.amax(d_i[0])-np.amin(d_i[0]))/2 + np.amin(d_i[0])
            s = splrep(d_t, d_i[0] - half_max)
            roots = sproot(s)
            fwhm = abs(roots[1]-roots[0])
            print(roots[1],roots[0],half_max)
            if flowrate>0:
                print("X-ray cell FWHMH =", fwhm, "ml")
            else:
                print("X-ray cell FWHMH =", fwhm, "min")
            ax1a.plot([roots[0], roots[1]],[half_max, half_max],"k-|")

        if calc_Rg and dkey=='subtracted':
            d_rg = np.asarray(d_rg)
            max_rg = np.max(d_rg)
            d_rg[d_rg==0] = np.nan
            ax1b.plot(d_rg, 'r.', label='rg')
            ax1b.set_xlim((0,len(d_rg)))
            ax1b.set_ylim((0, max_rg*1.05))
            ax1b.set_ylabel("Rg")
            leg = ax1b.legend(loc='center left', fontsize=9, frameon=False)
        else:
            ax1b.yaxis.set_major_formatter(plt.NullFormatter())

        if plot2d:
            if ax2 is None:
                ax2 = fig.add_axes(box2)
            ax2.tick_params(axis='x', top=True)
            ax2.xaxis.set_major_formatter(plt.NullFormatter())

            d2 = d_s + clim[0]/2
            ext = [0, len(data), len(self.qgrid), 0]
            asp = len(d_t)/len(self.qgrid)/(fig_w/fig_h1)
            im = ax2.imshow(d2, extent=ext, aspect="auto", norm=norm) 
            im.set_clim(clim)
            
            gpindex,gpvalues,gplabels = qgrid_labels(self.qgrid)
            ax2.set_yticks(gpindex)
            ax2.set_yticklabels(gplabels)
            ax2.set_ylabel('q')            
            
            ax2a = ax2.twinx()
            ax2a.set_ylim(len(self.qgrid)-1, 0)
            ax2a.set_ylabel('point #')
            

        #plt.tight_layout()
        #plt.show()
        
    def bin_subtracted_frames(self, sn=None, frame_range=None, first_frame=0, last_frame=-1, weighted=True,
                              plot_data=True, fig=None, qmax=0.5, qs=0.01,
                              save_data=False, path="", debug=False, txtWidget=None): 
        """ this is typically used after running subtract_buffer_SVD()
            the frames are specified by either first_frame and last_frame, or frame_range, e.g. "50-60"
            if path is used, be sure that it ends with '/'
        """
        sn = self.process_sample_name(sn, debug=debug)
        
        if plot_data:
            if fig is None:
                fig = plt.figure()            
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

        for fr in frame_range.strip(' ').split(','):    
            if fr is None:
                continue
            f1,f2 = fr.split('-')
            first_frame = int(f1)
            last_frame = int(f2)
            if last_frame<first_frame:
                last_frame=len(self.d1s[sn]['subtracted'])
            if debug is True:
                print(f"binning frames {fr}: first_frame={first_frame}, last_frame={last_frame}")
            d1s0 = copy.deepcopy(self.d1s[sn]['subtracted'][first_frame])
            if last_frame>first_frame+1:
                d1s0 = d1s0.avg(self.d1s[sn]['subtracted'][first_frame+1:last_frame], 
                                weighted=weighted, debug=debug)
            if save_data:
                d1s0.save(f"{path}{sn}_{first_frame:04d}-{last_frame-1:04d}s.dat", debug=debug)
            if plot_data:
                ax1.semilogy(d1s0.qgrid, d1s0.data)
                ax1.errorbar(d1s0.qgrid, d1s0.data, d1s0.err)
                ax1.set_xlim(0,qmax)
                i0,rg,_ = d1s0.plot_Guinier(qs=qs, ax=ax2)
                if txtWidget is not None:
                    txtWidget.value = f"I0={i0:.3g}, Rg={rg:.2f}"

        if plot_data:
            plt.tight_layout()   
            
        return d1s0
    
        
    def export_txt(self, sn=None, first_frame=0, last_frame=-1, save_subtracted=True,
                   averaging=False, plot_averaged=False, ax=None, path="",
                   debug=False):
        """ if path is used, be sure that it ends with '/'
        """
        sn = self.process_sample_name(sn, debug=debug)
        if save_subtracted:
            if 'subtracted' not in self.d1s[sn].keys():
                print("subtracted data not available.")
                return
            dkey = 'subtracted'
        else:
            if 'merged' not in self.d1s[sn].keys():
                print("1d data not available.")
                return
            dkey = 'merged'
        if last_frame<first_frame:
            last_frame=len(self.d1s[sn][dkey])

        d1s = self.d1s[sn][dkey][first_frame:last_frame]
        if averaging:
            d1s0 = copy.deepcopy(d1s[0])
            if len(d1s)>1:
                d1s0.avg(d1s[1:], weighted=True, plot_data=plot_averaged, ax=ax, debug=debug)
            d1s0.save(f"{path}{sn}_{first_frame:04d}-{last_frame-1:04d}{dkey[0]}.dat", 
                      debug=debug, footer=self.md_string(sn, md_keys=['HPLC']))
        else:
            for i in range(len(d1s)):
                d1s[i].save(f"{path}{sn}_{i+first_frame:04d}{dkey[0]}.dat", 
                            debug=debug, footer=self.md_string(sn, md_keys=['HPLC']))                    

