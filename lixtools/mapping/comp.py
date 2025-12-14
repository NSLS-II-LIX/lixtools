import pylab as plt
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import NMF,MiniBatchNMF

from .common import make_map_from_overall_attr


def estimate_Nc(x, mm, Ni=20, offset=0.1, cutoff=0.01):
    """ use SVD to estimate the number of eigen vectors needed to describe the dataset
        cutoff specifies the relative value of the lowest eigen value to be included (default 1% of the first)
    """
    V,S,U = randomized_svd(mm.T, Ni)
    print("SVD diagonal elements: ", S)
    eig_vectors = V*S
    coefs = U

    N = len(S[S>=cutoff*S[0]])
    
    fig, axs = plt.subplots(1,2,figsize=(9,5), gridspec_kw={'width_ratios': [2, 1]})
    for i in range(N):
        axs[0].plot(x, eig_vectors[:,i]-i*offset)
    axs[1].semilogy(S, "ko")
    axs[1].semilogy(N-1, S[N-1], "r.")
    
    return eig_vectors[:N]


def get_evs(x, mms, N=5, max_iter=5000, offset=0.1, use_minibatch=False, **kwargs):
    """ can be for multiple samples/datasets
        mms contains the background-subtracted data
    """
    if use_minibatch:
        model = MiniBatchNMF(n_components=N, max_iter=max_iter, **kwargs)
    else:
        model = NMF(n_components=N, max_iter=max_iter, **kwargs)
    W = model.fit_transform(np.vstack(mms).T)
    eig_vectors = W
    coefs = model.components_
    N = model.n_components_
    print(f"NMF stopped after {model.n_iter_} iterations, err = {model.reconstruction_err_}")

    plt.figure(figsize=(6,5))
    for i in range(eig_vectors.shape[1]):
        plt.plot(x, eig_vectors[:,i]-i*offset)    
    
    return eig_vectors,coefs,model

def make_ev_maps(dts, x, eig_vectors, coefs, res=None, name='q', abs_cor=False, template_grp="int_saxs"):    
    """ create maps f'ev{j}_{name}' based on decomposition of data into the given eigen vectors 
        decomposition results already given as coefs 
        residue could be included be saved in the data file, as f'res_{name}'
    """
    sl = 0
    N = eig_vectors.shape[-1]
    maps = [f'ev{i}_{name}' for i in range(N)]
    if res is not None:
        maps += [f'res_{name}']
    
    for i in range(len(dts)):
        dt = dts[i]        
        dt.load_data(read_data_keys=["attrs"], quiet=True)
        
        """
        for sn in dt.samples:
            ll = len(dt.proc_data[sn]['attrs'][template_grp])   #'transmission'
            for j in range(N):
                dt.proc_data[sn]['attrs'][f'ev{j}_{name}'] = coefs[j,sl:sl+ll]
            dt.proc_data[sn]['attrs'][f'res_{name}'] = res[sl:sl+ll]
            sl += ll

        dt.make_map_from_attr(attr_names=[f'ev{i}_{name}' for i in range(N)], correct_for_transsmission=abs_cor)
        dt.make_map_from_attr(attr_names=[f'res_{name}'], correct_for_transsmission=abs_cor)
        """
        
        ll = dt.proc_data['overall']['maps'][template_grp].d.size
        for j in range(N):
            make_map_from_overall_attr(dt, coefs[j,sl:sl+ll], template_grp=template_grp, 
                                       map_name=f'ev{j}_{name}', correct_for_transsmission=abs_cor)
        if res is not None:
            make_map_from_overall_attr(dt, res[sl:sl+ll], template_grp=template_grp, 
                                       map_name=f'res_{name}', correct_for_transsmission=abs_cor)
        sl += ll
        
        if not 'attrs' in dt.proc_data['overall'].keys():
            dt.proc_data['overall']['attrs'] = {}
        dt.proc_data['overall']['attrs'][f'evs_{name}'] = eig_vectors
        dt.proc_data['overall']['attrs'][f'ev_{name}'] = x
        dt.save_data(save_sns='overall', save_data_keys=['attrs'], save_sub_keys=[f'evs_{name}', f'ev_{name}'], quiet=True)
        dt.save_data(save_sns='overall', save_data_keys=['maps'], save_sub_keys=maps, quiet=True)
        
def check_ev_tomos(dt, ev_tag, ref_tomo='absorption', quiet=True):
    """ NOTE: OBSOLETE
              tomopy handles numberical values correctly; early versions of ganrec did not
              offset is due to incorrect rotation center
       
        scale the magnitude based on the values from the sinogram
        shift the tomo based on the ref_tomo
        these are necessary when the tomograms are reconstructed using different algorithms
    """
    grp = dt.proc_data['overall']
    tm0 = grp['tomo'][ref_tomo].d
    keys = [k for k in grp['tomo'].keys() if k[:2]=='ev' and ev_tag in k]
    
    for k in keys:
        tm1 = grp['tomo'][k].d*np.nansum(grp['maps'][k].d)/np.nansum(grp['tomo'][k].d)
        tm1[np.isnan(tm1)] = 0
        
        shift = pcc(tm0, tm1)[0]
        grp['tomo'][k].d = scipy.ndimage.shift(tm1, shift)
        
    dt.save_data(save_sns='overall', save_data_keys=['tomo'], save_sub_keys=keys, quiet=quiet)
    
def recombine(coef, method="nmf"):
    """ coef should have the same dimension as dt.proc_data['overall']['attrs'][f'evs_{method}'].shape[1]
    """
    N = dt.proc_data['overall']['attrs'][f'evs_{method}'].shape[1]
    if len(coef)!=N:
        raise Exception(f"shape mismatch: {len(coef)} != {N}")
        
    return np.sum(coef*dt.proc_data['overall']['attrs'][f'evs_{method}'], axis=1)
    

class ComponentSeparator:
    def __init__(self, datasets, qmask=None, datakey="Iq", subkey="subtracted"):
        """ datasets should be of the type h5xs_an
            each dataset should have the require data present, with the same qgrid
            use qmask to control which part of the scattering profile to be included in component separation
        """
        self.dts = datasets
        self.qgrid = datasets[0].get_h5_attr(f"overall/{datakey}/{subkey}", "qgrid")
        if qmask is None:
            qmask = [True for q in self.qgrid]
        self.qmask = qmask
        self.dk = datakey
        self.sk = subkey
        self.data1d = self.get_data()
    
    def get_data(self):
        return np.vstack([dt.proc_data['overall'][self.dk][self.sk][:,self.qmask] for dt in self.dts])

    def change_qmask(self, qmask):
        """ update the data used for component separation
        """
        self.qmask = qmask
        self.data1d = self.get_data()
    
    def estimate_N(self, Ni=20, offset=0.1, cutoff=0.01):
        estimate_Nc(self.qgrid[self.qmask], self.get_data(), Ni=Ni, offset=offset, cutoff=cutoff)

    def get_evs(self, N, max_iter=5000, offset=0.1, use_minibatch=False, **kwargs):
        """ eig_vectors,coefs,model are returned
        """
        self.evs,self.coefs,self.model = get_evs(self.qgrid[self.qmask], self.data1d, N=N, max_iter=max_iter, 
                                                 offset=offset, use_minibatch=use_minibatch, **kwargs)
        self.res = np.fabs(np.dot(self.evs, self.coefs).T-self.data1d)

    def show_overall_residue(self, thresh=1e-3):
        plt.figure()
        dd = np.sum(self.data1d, axis=1)
        rr = np.sqrt(np.sum(self.res**2, axis=1))
        idx = (dd>thresh)
        pp[idx] /= rr[idx]
        pp[~idx] = np.nan
        plt.plot(pp)

    def show_single_point_residue(self, i):
        qq = self.qgrid[self.qmask]
        plt.figure()
        plt.plot(qq, self.data1d[i])
        plt.plot(qq, self.res[i])
        plt.plot(qq, np.dot(self.evs, self.coefs[:,i]))
    
    def make_ev_maps(self, label):
        """ add a label to distinguish between different eigenvectors
        """
        self.dts[0].load_data("overall", ["maps"])
        tgrp = self.dts[0].proc_data['overall']['maps'].keys()[0]
        make_ev_maps(self.dts, self.qgrid[self.qmask], self.evs, self.coefs, res=self.res, 
                     name=label, abs_cor=False, template_grp=tgrp)
        
    def clustering_analysis(self):
        pass