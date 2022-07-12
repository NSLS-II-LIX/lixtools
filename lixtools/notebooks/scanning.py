import ipywidgets
from IPython.display import display,clear_output
import numpy as np
import json,os
from py4xs.data2d import Data2d,unflip_array,flip_array
from py4xs.hdf import lsh5
from py4xs.plot import show_data
from lixtools.hdf import h5xs_an,h5xs_scan
import pylab as plt

scan_GUI_par = {
    'roi_qrange': '0.1~0.15,1.0~1.1,1.58~1.63',
    'roi_phirange': '-180~180'
}

def get_range(tx, alt_values, prec=2):
    """ should be of the format v1~v2, v1~v2
        if successful, return float values [[v1, v2], ...]
        otherwise reture alt_values for each pair
    """
    ret = []
    tt = []
    
    for v in tx.value.split(","):
        xx = v.split("~")
        if len(xx)==2:
            v1 = float(xx[0])
            v2 = float(xx[1])
            tt.append(v)
            ret.append([v1,v2])
        else:
            tt.append(f"{alt_values[0]:.{prec}f}-{alt_values[1]:.{prec}f}")
            ret.append(alt_values)
    
    tx.value = ",".join(tt)
    return ret
            
def display_data_scanning(dt):
    """ the format of qrange and phirange should be v1~v2, v1~v2
    """
    sn = list(dt.proc_data.keys())[0]
    dks = list(dt.proc_data[sn].keys())
    sks = list(dt.proc_data[sn][dks[0]].keys())
    ddSample = ipywidgets.Dropdown(options=list(dt.fh5.keys()), value=sn, description='Sample:')
    ddDataKey = ipywidgets.Dropdown(options=dks, value=dks[0], description='Data Key:')
    ddSubKey = ipywidgets.Dropdown(options=sks, value=sks[0], description='Sub-Key:')
    hbox1 = ipywidgets.HBox([ddSample, ddDataKey, ddSubKey])
    
    slideFrn = ipywidgets.IntSlider(value=0, min=0, max=100, step=1, description='frame #:',
                                    disabled=False, continuous_update=False, orientation='horizontal', 
                                    readout=True, readout_format='d')
    ddScale = ipywidgets.Dropdown(options=["none", "x", "x1.5", "x2"], value="none", description='multiplier:')
    symmCbox = ipywidgets.Checkbox(value=False, description='apply symmetry', 
                                layout=ipywidgets.Layout(width='17%'))
    qrangeTx = ipywidgets.Text(value=scan_GUI_par['roi_qrange'], description='q range:', 
                               layout=ipywidgets.Layout(width='13%'))       
    phirangeTx = ipywidgets.Text(value=scan_GUI_par['roi_phirange'], description='phi range:', 
                                 layout=ipywidgets.Layout(width='13%'))       
    hbox2 = ipywidgets.HBox([slideFrn, ddScale, symmCbox, qrangeTx, phirangeTx])
        
    logCbox = ipywidgets.Checkbox(value=False, description='logScale', 
                                layout=ipywidgets.Layout(width='14%'))
    txOutput = ipywidgets.Textarea(value='', placeholder='', description='Message:', disabled=False, 
                                   layout=ipywidgets.Layout(width='50%'))
    hbox3 = ipywidgets.HBox([logCbox, txOutput])
    
    box = ipywidgets.VBox([hbox1, hbox2, hbox3])
                           
    display(box)    
    fig = plt.figure(figsize=(10,6))
    
    def updatePlots(w):
        sn = ddSample.value
        dk = ddDataKey.value
        sk = ddSubKey.value
        txOutput.value = f"updating ... {sn},{dk},{sk} "
        if not dk in dt.proc_data[sn].keys():
            return
        if not sk in dt.proc_data[sn][dk].keys():
            return        
        
        fig.clear()
        if dk=="qphi":
            frn = slideFrn.value
            ax1 = fig.add_axes([0.1, 0.5, 0.5, 0.4])
            ax2 = fig.add_axes([0.1, 0.1, 0.5, 0.25])
            ax3 = fig.add_axes([0.75, 0.5, 0.2, 0.4])
            try:
                dm = dt.proc_data[sn][dk][sk][frn]
            except:
                txOutput.value = f"{sn}, {dk}, {sk}, {frn}"
                
            if symmCbox.value:
                dm = dm.apply_symmetry()
            sc_factor = ddScale.value
            if sc_factor=="none":
                sc_factor = None
            dm.plot(ax=ax1, logScale=logCbox.value, sc_factor=sc_factor) 
            
            phi_range = get_range(phirangeTx, [dm.yc[0], dm.yc[-1]], 0)[0]
            scan_GUI_par['roi_phirange'] = phirangeTx.value
            q_ranges = get_range(qrangeTx, [dm.xc[0], dm.xc[-1]], 2)
            scan_GUI_par['roi_qrange'] = qrangeTx.value
            _,_,_ = dm.line_profile("x", plot_data=True, yrange=phi_range, ax=ax2)
            for qrng in q_ranges:
                _,_,_ = dm.line_profile("y", plot_data=True, xrange=qrng, ax=ax3) 
            
            txOutput.value = f"displaying frn # {frn}"
            if logCbox.value:
                ax2.set_yscale('log')
                ax2.set_xscale('log')
                ax3.set_yscale('log')
            ax3.set_ylim(bottom=0)
        else:
            if dk=="attrs":  # 1D array
                ax1 = fig.add_axes([0.1, 0.5, 0.5, 0.4])
                ax1.plot(dt.proc_data[sn][dk][sk])
            elif dk in ["maps","tomo"]:  # MatrixWithCoords
                ax1 = fig.add_axes([0.1, 0.1, 0.5, 0.8])
                dt.proc_data[sn][dk][sk].plot(ax=ax1, logScale=logCbox.value)
            elif dk=="avg_data":
                d2s = {}
                for ext,img in dt.proc_data[sn]['avg_data'].items():
                    det = dt.det(ext)
                    ep = det.exp_para
                    d2 = Data2d(unflip_array(img, ep.flip), exp=ep)
                    d2s[ext] = d2
                    d2.md['frame #'] = "average"
                show_data(d2s, fig=fig, aspect=1, logScale=logCbox.value, showMask=False, cmap='jet')
        
        txOutput.value += f"\ndone .."
        
        plt.ion()
    
    def onChangeSample(w):
        if w['name']!='value':
            return
        sn = ddSample.value
        dk = ddDataKey.value
        klist = list(dt.proc_data[sn].keys())
        ddDataKey.options = klist
        if not dk in klist:
            ddDataKey.value = klist[-1]
        else:
            txOutput.value = f"sample changed: {sn} \n"
            updatePlots(None)
    
    def onChangeDataKey(w):
        if w['name']!='value':
            return
        sn = ddSample.value
        dk = ddDataKey.value
        sk = ddSubKey.value
        klist = list(dt.proc_data[sn][dk].keys())
        ddSubKey.options = klist
        if not sk in klist:
            ddSubKey.value = klist[-1]
        else: 
            txOutput.value += f"data key changed: {dk} \n"
            updatePlots(None)
            
    def onChangeSubKey(w):
        if w['name']!='value':
            return
        sn = ddSample.value
        dk = ddDataKey.value
        if not dk in dt.proc_data[sn].keys():
            return
        sk = ddSubKey.value
        if not sk in dt.proc_data[sn][dk].keys():
            return
        txOutput.value += f"subkey changed {sk} \n"
        try:
            nn = len(dt.proc_data[sn][dk][sk])
            if nn>1:
                slideFrn.max = nn-1
                slideFrn.disabled=False
            else:
                slideFrn.disabled=True
        except:
            slideFrn.disabled=True
        updatePlots(None)
                        
    updatePlots(None)
    ddSample.observe(onChangeSample)
    ddDataKey.observe(onChangeDataKey)
    ddSubKey.observe(onChangeSubKey)
    slideFrn.observe(updatePlots)
    logCbox.observe(updatePlots)
    symmCbox.observe(updatePlots)
    
    return dt