import ipywidgets
from IPython.display import display,clear_output
import numpy as np
import json,os
from py4xs.slnxs import trans_mode
from py4xs.slnxs import get_font_size
from py4xs.hdf import h5xs,lsh5,create_linked_files
from lixtools.hdf import h5sol_HT,h5sol_HPLC
from lixtools.atsas import gen_atsas_report
import pylab as plt
from scipy import interpolate,integrate

                  
#def display_data_h5xs(fn1, fn2=None, field='merged', trans_field = 'em2_sum_all_mean_value'):
def display_data_h5xs(fns, field='merged', trans_field = 'em2_sum_all_mean_value'):

    def onChangeSample(w):
        sel1 = [sampleLabels[i] for i in range(len(sampleLabels)) 
                if dt1.attrs[ddSample.value]['selected'][i]]
        smAverageSM.value = sel1  
        updateAvgPlot(None)
        
    def onChangeBlank(w):
        sel2 = [blankLabels[i] for i in range(len(blankLabels)) 
                if dt1.attrs[ddBlank.value]['selected'][i]]   # dt2
        blAverageSM.value = sel2    
        updateAvgPlot(None)
        
    def updateAvgPlot(w):
        ax01.clear()
        sn1 = ddSample.value
        sel1 = [(sampleLabels[i] in smAverageSM.value) for i in range(len(sampleLabels))]
        d1a1 = avg_d1(dt1.d1s[sn1][field], sel1, ax01)
        if np.any(sel1 != dt1.attrs[sn1]['selected']):        
            dt1.attrs[sn1]['selected'] = sel1
            
        ax02.clear()
        sn2 = ddBlank.value
        sel2 = [(blankLabels[i] in blAverageSM.value) for i in range(len(blankLabels))]
        d1a2 = avg_d1(dt1.d1s[sn2][field], sel2, ax02)  # dt2
        if np.any(sel2 != dt1.attrs[sn2]['selected']):        
            dt1.attrs[sn2]['selected'] = sel2   #dt2
            
        ax03.clear()
        if d1a1 is not None and d1a2 is not None:
            d1fb = d1a1.bkg_cor(d1a2, plot_data=True, ax=ax03, 
                                sc_factor=ftScale1.value, debug='quiet')
        elif d1a1 is not None:
            d1fb = d1a1
            d1fb.plot(ax=ax03)
        
        return d1fb
            
    def save_d1s(w):
        """ should update the selection field in h5 file
            add d1s to a list
        """
        d1fb = updateAvgPlot(None)
        sn1 = ddSample.value
        sn2 = ddBlank.value
        dt1.fh5[f'{sn1}/processed'].attrs['selected'] = dt1.attrs[sn1]['selected']
        dt1.fh5.flush()
        #dt2.fh5[f'{sn2}/processed'].attrs['selected'] = dt2.attrs[sn2]['selected']
        #dt2.fh5.flush()
        if sn1 in d1list.keys():
            del d1list[sn1]
        d1list[sn1] = d1fb
        
        ddSampleS.index = None
        ddSampleS.options = list(d1list.keys())
        ddSolventS.index = None
        ddSolventS.options = ['None'] + list(d1list.keys()) 
        
    def onUpdatePlot(w):
        if ddSampleS.value is None:
            return

        ax.clear()
        d1s = d1list[ddSampleS.value]
        if ddSolventS.value in [None, 'None', d1list[ddSampleS.value]]:
            d1f = d1s
            d1f.plot(ax=ax)
        else:
            d1b = d1list[ddSolventS.value]
            d1f = d1s.bkg_cor(d1b, plot_data=True, ax=ax, 
                              sc_factor=slideScFactor.value, debug='quiet')
        return d1f
        
    def onExport(w):
        sn = ddSampleS.value
        fn = f"{sn}_{txFnSuffix.value}.dat"
        d1f = onUpdatePlot(None)
        d1f.save(fn)
        
    def avg_d1(d1s, selection, ax):
        d1sl = [d1s[i] for i in range(len(selection)) if selection[i]]
        if len(d1sl)==0:
            return None
        else:
            return d1sl[0].avg(d1sl[1:], plot_data=True, ax=ax, debug='quiet')
    
    if isinstance(fns, str):
        fn = fns
    elif not isinstance(fns, list):
        raise Exception(f"input is not a filename or a list of filenames: {fns}")
    elif len(fns)==1:
        fn = fns[0]
    else:
        # create a temporary file to link to the individual files
        fn = "t.h5"
        create_linked_files(fn, fns)
    
    dt1 = h5xs(fn, transField=trans_field)
    dt1.load_d1s() 
    dt1.set_trans(trans_mode.external)
    dt1.average_d1s(filter_data=True, debug="quiet")
    #dt2 = dt1
              
    d1list = {}

    fields = list(set(dt1.d1s[dt1.samples[0]].keys())-set(['averaged']))
    if field not in fields:
        print(f"invalid field, options are {fields}.")
    
    # widgets
    ddSample = ipywidgets.Dropdown(options=dt1.samples, value=dt1.samples[0], description='Sample:')
    sampleLabels = [f"frame #{i}" for i in range(len(dt1.attrs[dt1.samples[0]]['selected']))]
    smAverageSM = ipywidgets.SelectMultiple(options=sampleLabels, descripetion="selection for averaging")

    vbox1 = ipywidgets.VBox([ddSample, smAverageSM])                
    
    ddBlank = ipywidgets.Dropdown(options=dt1.samples, value=dt1.samples[-1], description='Blank:')
    blankLabels = [f"frame #{i}" for i in range(len(dt1.attrs[dt1.samples[-1]]['selected']))]
    blAverageSM = ipywidgets.SelectMultiple(options=blankLabels, descripetion="selection for averaging")
    vbox2 = ipywidgets.VBox([ddBlank, blAverageSM])        
    
    btnUpdate = ipywidgets.Button(description='Update plot')
    btnSave1D = ipywidgets.Button(description='Save 1D')
    ftScale1 = ipywidgets.FloatText(value=0.998, description='blank scale:', disabled=False)
    vbox3 = ipywidgets.VBox([btnUpdate, btnSave1D, ftScale1])
    
    hbox1 = ipywidgets.HBox([vbox1, vbox2, vbox3])

    fig = plt.figure(figsize=(9,4))
    ax01 = fig.add_axes([0.1, 0.15, 0.23, 0.8])
    ax02 = fig.add_axes([0.4, 0.15, 0.23, 0.8])
    ax03 = fig.add_axes([0.7, 0.15, 0.23, 0.8])
    
    ddSampleS = ipywidgets.Dropdown(description='Sample:')
    ddSolventS = ipywidgets.Dropdown(description='Solvent:')    
    hbox2 = ipywidgets.HBox([ddSampleS, ddSolventS])  

    slideScFactor = ipywidgets.FloatSlider(value=1.0, min=0.2, max=5.0, step=0.0001,
                                           description='Scaling factor:', readout_format='.4f')
    btnExport = ipywidgets.Button(description='Export')
    txFnSuffix = ipywidgets.Text(value='s', description='filename suffix:', disabled=False, 
                                 layout=ipywidgets.Layout(width='10%'))
    hbox3 = ipywidgets.HBox([slideScFactor, btnExport, txFnSuffix])

    box = ipywidgets.VBox([ipywidgets.Label(value="___ Blank subtraction: ___"), 
                           hbox1, ipywidgets.Label(value="___ After blank subtraction: ___"), 
                           hbox2, hbox3])  
    display(box)
    fig = plt.figure(figsize=(7,5))
    ax = plt.gca()

    onChangeSample(None)
    onChangeBlank(None)
    btnUpdate.on_click(updateAvgPlot)
    slideScFactor.observe(onUpdatePlot)
    ddSample.observe(onChangeSample)
    ddBlank.observe(onChangeBlank)
    btnExport.on_click(onExport)
    btnSave1D.on_click(save_d1s)
    
    return dt1

