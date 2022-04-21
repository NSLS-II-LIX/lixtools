import ipywidgets
from IPython.display import display,clear_output
import numpy as np
from lixtools.hdf import h5sol_HT,h5sol_HPLC
import pylab as plt

from .subtract_buffer_mcr import subtract_buffer_mcr
from .sol_hplc import HPLC_GUI_par
    
def prep_data(dt, show_data=False):
    dd2s = np.vstack([d1.data for d1 in dt.d1s[dt.samples[0]]['subtracted']]).T
    qgrid = dt.qgrid

    dd2s.shape, len(qgrid)

    d1 = np.min(dd2s, axis=1)
    m1 = (d1>0)
    m1 = m1 & (np.append(m1[1:],True)) & (np.insert(m1[:-1],0,True))

    if show_data:
        plt.figure(figsize=(7,3))
        plt.subplot(121)
        plt.imshow(np.log(dd2s))
        plt.subplot(122)
        plt.imshow(np.log(np.compress(m1, dd2s, axis=0)))

    return np.compress(m1, dd2s, axis=0), qgrid[m1]

def performMCR(dt: h5sol_HPLC):

    peakGuessTx = ipywidgets.Text(value=HPLC_GUI_par['peak_guess'],
                                  description='peak guess frames:',
                                  layout=ipywidgets.Layout(width='60%'),
                                  style={'description_width': 'initial'})
    halfWidthTx = ipywidgets.Text(value=HPLC_GUI_par['half_width'],
                                  description='max half width:',
                                  layout=ipywidgets.Layout(width='40%'),
                                  style={'description_width': 'initial'})
    frnsSubTx = ipywidgets.Text(value=HPLC_GUI_par['frns_sub'], 
                                    description='buffer frames:', 
                                    layout=ipywidgets.Layout(width='40%'),
                                    style = {'description_width': 'initial'})
    guinierQTx = ipywidgets.Text(value=HPLC_GUI_par['guinier_q'],
                                 description='Guinier q ranges:',
                                 layout=ipywidgets.Layout(width='60%'),
                                 style={'description_width': 'initial'})
    gradThreshTx = ipywidgets.Text(value=HPLC_GUI_par['grad_thresh'],
                                   description='gradient thresh:',
                                   layout=ipywidgets.Layout(width='40%'),
                                   style={'description_width': 'initial'})

    optMet1Dd = ipywidgets.Dropdown(options=['dogbox', 'trf', 'lm'],
                                    value=HPLC_GUI_par['opt_method_step1'],
                                    description='Step 1 Opt Method:',
                                    layout=ipywidgets.Layout(width='50%'),
                                    style={'description_width': 'initial'})
    optMet2Dd = ipywidgets.Dropdown(options=['dogbox', 'trf', 'lm'],
                                    value=HPLC_GUI_par['opt_method_step2'],
                                    description='Step 2 Opt Method:',
                                    layout=ipywidgets.Layout(width='50%'),
                                    style={'description_width': 'initial'})

    hbox321c = ipywidgets.HBox([peakGuessTx, halfWidthTx, frnsSubTx])
    hbox322c = ipywidgets.HBox([guinierQTx, gradThreshTx])
    hbox32c = ipywidgets.VBox([hbox321c, hbox322c])
    hbox32d = ipywidgets.HBox([optMet1Dd, optMet2Dd])
    
    txtbox = ipywidgets.Textarea(layout=ipywidgets.Layout(width='90%'))

    btnRun = ipywidgets.Button(description='Run')
    btn2dplot = ipywidgets.Button(description='Plot 2D')

    fig2 = plt.figure(figsize=(9,6))
    HPLC_GUI_par["mcr_ret"] = None
    
    dd2s,qgrid = prep_data(dt)
    
    def plot2d(w):
        if HPLC_GUI_par["mcr_ret"] is None:
            txtbox.value = "run MCR first."
            return
        fig2.clear()
        ax1 = fig2.add_subplot(121)
        ax2 = fig2.add_subplot(122)
        fig2.subplots_adjust(hspace=0.4)
        ax1.imshow(np.log(dd2s))
        cprof,bv = HPLC_GUI_par["mcr_ret"]
        dd2s1 = (cprof[1:].T @ bv[1:]).T
        ax2.imshow(np.log(dd2s1))
        
    def run(w):
        fig2.clear()
        ax1_xs = fig2.add_subplot(221)
        ax1_conc = fig2.add_subplot(222)
        ax2_xs = fig2.add_subplot(223)
        ax2_conc = fig2.add_subplot(224)
        fig2.subplots_adjust(hspace=0.4, wspace=0.3)
        txtbox.value = "working ..."
        try:
            cprof,bv = subtract_buffer_mcr(dd2s, qgrid, 
                peak_pos_guess=peakGuessTx.value,
                max_half_width=halfWidthTx.value,
                iframe_bg=int(frnsSubTx.value.split('-')[1]),
                guinier_q_ranges=guinierQTx.value,
                grad_threshes=gradThreshTx.value,
                opt_methods=(optMet1Dd.value, optMet2Dd.value),
                ax1_xs=ax1_xs,
                ax1_conc=ax1_conc,
                ax2_xs=ax2_xs,
                ax2_conc=ax2_conc)
        except Exception as e:
            txtbox.value = str(e)
        else:
            txtbox.value = "completed ...  "
            HPLC_GUI_par["mcr_ret"] = [cprof, bv]
        
    display(ipywidgets.VBox([hbox32c, hbox32d,  
                             ipywidgets.HBox([btnRun, btn2dplot]), txtbox], 
                            layout=ipywidgets.Layout(width='90%')))
    
    btnRun.on_click(run)
    btn2dplot.on_click(plot2d)