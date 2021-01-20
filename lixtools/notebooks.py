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

def display_solHT_data(fn, atsas_path=""):
    """ atsas_path for windows might be c:\atsas\bin
    """
    dt = h5sol_HT(fn)
    dt.load_d1s()
    dt.subtract_buffer(sc_factor=-1, debug='quiet')
    if not os.path.exists("processed/"):
        os.mkdir("processed")
    
    if "run_type" not in dt.fh5['/'].attrs.keys():
        dt.fh5['/'].attrs['run_type'] = 'static'
        dt.fh5['/'].attrs['instrument'] = 'LiX'
        dt.fh5.flush()
    elif dt.fh5['/'].attrs['run_type']!='static':
        raise Exception(f"this h5 has been assigned an incompatible run_type: {dt.fh5['/'].attrs['run_type']}")
    
    # widgets
    ddSample = ipywidgets.Dropdown(options=dt.samples, 
                                   value=dt.samples[0], description='Sample:')
    sampleLabels = [f"frame #{i}" for i in range(len(dt.attrs[dt.samples[0]]['selected']))]
    smAverage = ipywidgets.SelectMultiple(options=sampleLabels, 
                                          layout=ipywidgets.Layout(width='10%'),
                                          descripetion="selection for averaging")
    
    btnExport = ipywidgets.Button(description='Export', 
                                  layout=ipywidgets.Layout(width='20%'), 
                                  style = {'description_width': 'initial'})
    exportSubtractedCB = ipywidgets.Checkbox(value=True, description='export subtracted',
                                             #layout=ipywidgets.Layout(width='30%'),
                                             style = {'description_width': 'initial'})
    btnUpdate = ipywidgets.Button(description='Update plot', layout=ipywidgets.Layout(width='35%'))

    subtractCB = ipywidgets.Checkbox(value=False, 
                                     style = {'description_width': 'initial'},
                                     description='show subtracted')

    slideScFactor = ipywidgets.FloatSlider(value=1.0, min=0.8, max=1.2, step=0.0001,
                                           style = {'description_width': 'initial'},
                                           description='Scaling factor:', readout_format='.4f')
    guinierQsTx = ipywidgets.Text(value='0.01', 
                                  layout=ipywidgets.Layout(width='40%'),
                                  description='Guinier fit qs:')
    guinierRgTx = ipywidgets.Text(value='', 
                                  layout=ipywidgets.Layout(width='35%'), 
                                  description='Rg:')

    vbox1 = ipywidgets.VBox([ddSample, slideScFactor, 
                             ipywidgets.HBox([guinierQsTx, guinierRgTx])],
                            layout=ipywidgets.Layout(width='40%'))
    vbox2 = ipywidgets.VBox([ipywidgets.HBox([btnExport, exportSubtractedCB]), 
                             btnUpdate, subtractCB], 
                            layout=ipywidgets.Layout(width='45%'))
    hbox1 = ipywidgets.HBox([vbox1, smAverage, vbox2])                

    btnReport = ipywidgets.Button(description='ATSAS report') #, layout=ipywidgets.Layout(width='20%'))
    qSkipTx = ipywidgets.Text(value='0', description='skip:', 
                                layout=ipywidgets.Layout(width='20%'),
                                style = {'description_width': 'initial'})    
    qCutoffTx = ipywidgets.Text(value='0.3', description='q cutoff:', 
                                layout=ipywidgets.Layout(width='25%'),
                                style = {'description_width': 'initial'})    
    outTxt = ipywidgets.Textarea(layout=ipywidgets.Layout(width='55%', height='100%'))
    hbox5 = ipywidgets.HBox([outTxt, 
                             ipywidgets.VBox([btnReport, 
                                              ipywidgets.HBox([qSkipTx, qCutoffTx])]) ])    
    
    box = ipywidgets.VBox([hbox1, hbox5])
    display(box)
    fig1 = plt.figure(figsize=(7, 4))
    # rect = l, b, w, h
    ax1 = fig1.add_axes([0.1, 0.15, 0.5, 0.78])
    ax2 = fig1.add_axes([0.72, 0.61, 0.26, 0.32])
    ax3 = fig1.add_axes([0.72, 0.15, 0.26, 0.32])

    axr = []
    fig2 = plt.figure(figsize=(7,2.5))
    axr.append(fig2.add_axes([0.09, 0.25, 0.25, 0.6])) 
    axr.append(fig2.add_axes([0.41, 0.25, 0.25, 0.6])) 
    axr.append(fig2.add_axes([0.73, 0.25, 0.25, 0.6])) 
    axr.append(axr[0].twiny())
    
    def onChangeSample(w):
        sn = ddSample.value
        sel = [sampleLabels[i] for i in range(len(sampleLabels)) 
               if dt.attrs[sn]['selected'][i]]
        smAverage.value = sel    
        isSample = ('sc_factor' in dt.attrs[sn].keys())
        for a in axr:
            a.clear()
        outTxt.value = ""

        if isSample:
            subtractCB.disabled = False
            slideScFactor.value = dt.attrs[sn]['sc_factor']
            exportSubtractedCB.disabled = False
            if subtractCB.value:
                btnReport.disabled = False
        else:
            subtractCB.value = False
            subtractCB.disabled = True
            slideScFactor.disabled = True
            exportSubtractedCB.value = False
            exportSubtractedCB.disabled = True
            btnReport.disabled = True
        onUpdatePlot(None)
    
    def onReport(w):
        #try:
        txt = gen_atsas_report(dt.d1s[ddSample.value]["subtracted"], ax=axr, sn=ddSample.value,
                               skip=int(qSkipTx.value), q_cutoff=float(qCutoffTx.value), 
                               print_results=False, path=atsas_path)
        outTxt.value = txt
        #except:
        #    outTxt.value = "unable to run ATSAS ..."
    
    def onUpdatePlot(w):
        sn = ddSample.value
        re_calc = False
        show_sub = subtractCB.value
        sc_factor = slideScFactor.value
        sel = [(sampleLabels[i] in smAverage.value) for i in range(len(sampleLabels))]
        isSample = ('sc_factor' in dt.attrs[sn].keys())
        if w is not None:
            if np.any(sel != dt.attrs[sn]['selected']):
                dt.average_d1s(sn, selection=sel, debug=False)
                if isSample:
                    re_calc = True
            if isSample:
                if sc_factor!=dt.attrs[sn]['sc_factor']:
                    re_calc = True
            if re_calc:
                dt.subtract_buffer(sn, sc_factor=sc_factor, debug='quiet')
                re_calc = False
        ax1.clear()
        dt.plot_sample(sn, ax=ax1, show_subtracted=show_sub)
        ax2.clear()
        ax3.clear()
        if isSample and show_sub:
            d1 = dt.d1s[sn]['subtracted']
            ym = np.max(d1.data[d1.qgrid>0.5])
            qm = d1.qgrid[d1.data>0][-1]
            ax2.semilogy(d1.qgrid, d1.data)
            #ax2.errorbar(d1.qgrid, d1.data, d1.err)
            ax2.set_xlim(left=0.5, right=qm)
            ax2.set_ylim(top=ym*1.1)
            ax2.yaxis.set_major_formatter(plt.NullFormatter())
            qs = np.float(guinierQsTx.value)
            i0,rg,_ = dt.d1s[sn]['subtracted'].plot_Guinier(ax=ax3, qs=qs, fontsize=0)
            ax3.yaxis.set_major_formatter(plt.NullFormatter())
            guinierRgTx.value = ("%.2f" % rg)
            #print(f"I0={i0}, Rg={.2f:rg}")
            #plt.tight_layout()
            ax2.set_title("buf subtraction")
            ax3.set_title("Guinier")
    
    def onShowSubChanged(w):
        show_sub = subtractCB.value
        if show_sub:
            slideScFactor.disabled = False
            smAverage.disabled = True
            btnReport.disabled = False
        else:
            slideScFactor.disabled = True
            smAverage.disabled = False
            btnReport.disabled = True
        onUpdatePlot(None)
    
    def onExport(w):
        sn = ddSample.value
        dt.export_d1s(sn, path="processed/", save_subtracted=exportSubtractedCB.value)
        dt.update_h5()
        
    onChangeSample(None)
    btnUpdate.on_click(onUpdatePlot)
    subtractCB.observe(onShowSubChanged)
    slideScFactor.observe(onUpdatePlot)
    ddSample.observe(onChangeSample)
    btnExport.on_click(onExport)
    btnReport.on_click(onReport)
    
    return dt

                  
#def display_data_h5xs(fn1, fn2=None, field='merged', trans_field = 'em2_sum_all_mean_value'):
def display_data_h5xs(fns, field='merged', trans_field = 'em2_sum_all_mean_value'):

    def onChangeSample(w):
        sel1 = [sampleLabels[i] for i in range(len(sampleLabels)) 
                if dt1.attrs[ddSample.value]['selected'][i]]
        smAverageSM.value = sel1  
        updateAvgPlot(None)
        
    def onChangeBlank(w):
        sel2 = [blankLabels[i] for i in range(len(blankLabels)) 
                if dt2.attrs[ddBlank.value]['selected'][i]]
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
        d1a2 = avg_d1(dt2.d1s[sn2][field], sel2, ax02)
        if np.any(sel2 != dt1.attrs[sn2]['selected']):        
            dt2.attrs[sn2]['selected'] = sel2
            
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
        dt2.fh5[f'{sn2}/processed'].attrs['selected'] = dt2.attrs[sn2]['selected']
        dt2.fh5.flush()
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
        
    def set_trans(dh5, field, trans_field):
        """ at LiX, the transmitted beam intensity (beam stop, em2_sum_all_mean_value) can be recorded
            in two ways, the location of the data varies accordingly: (A) em2 as a detector, data under 
            the "primary" stream, (B) em2 as a monitor, data under the 
        """
        sn = dh5.samples[0]
        stlist = [_ for _ in list(dh5.fh5[sn].keys()) if trans_field in _]
        if len(stlist)>0:
            stream_name = stlist[0]   
            for sn in dh5.samples:
                # first interpolate the monitor counts
                h = json.loads(dh5.fh5[sn].attrs["start"])
                di = dh5.fh5[f"{sn}/{stream_name}/data/{trans_field}"][...]
                dt = dh5.fh5[f"{sn}/{stream_name}/timestamps/{trans_field}"][...]
                # this will interpolate the em values, maybe unnecessary, especially if em values
                # are read more frequently than detector exposures 
                #fi = interpolate.interp1d(dt, di)
                if h['plan_name']=="raster":
                    fast_axis = h["motors"][-1]
                    dt1 = dh5.fh5[f"{sn}/primary/timestamps/{fast_axis}"][...].flatten()
                    exp = dt1[1]-dt1[0]  # the exposure time needs to be added to the header 
                    for i in range(len(dh5.d1s[sn][field])):
                        t = dt1[i] # this is the time stamp on the trigger
                        #trans = integrate.quad(fi, t, t+exp)[0]/exp
                        ti1 = np.fabs(t-dt).argmin()
                        ti2 = np.fabs(dt-(t+exp)).argmin()
                        if ti2==ti1:
                            trans = di[ti1]
                        else:
                            trans = np.sum(di[ti1:ti2])/(ti2-ti1)
                        dh5.d1s[sn][field][i].set_trans(trans, transMode=trans_mode.external)
                else:
                    raise Exception(f"don't know how to handle plan: {h['plan_name']}")
        else:
            stream_name = "primary"      
            for sn in dh5.samples:
                for i in range(len(dh5.d1s[sn][field])):
                    dh5.d1s[sn][field][i].set_trans(dh5.fh5[f'{sn}/{stream_name}/data/{trans_field}'][i], 
                                                transMode=trans_mode.external)

    def avg_d1(d1s, selection, ax):
        d1sl = [d1s[i] for i in range(len(selection)) if selection[i]]
        if len(d1sl)==0:
            return None
        else:
            return d1sl[0].avg(d1sl[1:], plot_data=True, ax=ax, debug='quiet')
    
    #dt1 = h5xs(fn1)
    #dt1.load_d1s()
    #set_trans(dt1, field, trans_field)
    #if fn2 is not None:
    #    dt2 = h5xs(fn2)
    #    dt2.load_d1s()    
    #    set_trans(dt2, field, trans_field)
    #else:
    #    dt2 = dt1
    
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
    dt1 = h5xs(fn)
    dt1.load_d1s() 
    set_trans(dt1, field, trans_field)
    dt2 = dt1
              
    d1list = {}

    fields = list(set(dt1.d1s[dt1.samples[0]].keys())-set(['averaged']))
    if field not in fields:
        print(f"invalid field, options are {fields}.")
    
    # widgets
    ddSample = ipywidgets.Dropdown(options=dt1.samples, value=dt1.samples[0], description='Sample:')
    sampleLabels = [f"frame #{i}" for i in range(len(dt1.attrs[dt1.samples[0]]['selected']))]
    smAverageSM = ipywidgets.SelectMultiple(options=sampleLabels, descripetion="selection for averaging")

    vbox1 = ipywidgets.VBox([ddSample, smAverageSM])                
    
    ddBlank = ipywidgets.Dropdown(options=dt2.samples, value=dt2.samples[0], description='Blank:')
    blankLabels = [f"frame #{i}" for i in range(len(dt2.attrs[dt2.samples[0]]['selected']))]
    blAverageSM = ipywidgets.SelectMultiple(options=blankLabels, descripetion="selection for averaging")
    vbox2 = ipywidgets.VBox([ddBlank, blAverageSM])        
    
    btnUpdate = ipywidgets.Button(description='Update plot')
    btnSave1D = ipywidgets.Button(description='Save 1D')
    ftScale1 = ipywidgets.FloatText(value=0.998, description='blank scale:', disabled=False)
    vbox3 = ipywidgets.VBox([btnUpdate, btnSave1D, ftScale1])
    
    hbox1 = ipywidgets.HBox([vbox1, vbox2, vbox3])

    fig = plt.figure(figsize=(12,4))
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

              
HPLC_GUI_par = {
    'xroi1': '0.02, 0.03',
    'xroi2': '',
    'xroi3': '',
    'xroi_log': False,
    'show_LC1': True,
    'show_LC2': False,
    'show_2d_log': True,
    'show_2d_sub': False,
    'cmin': '0.01',
    'cmax': '200',
    'sub_mode': 'normal',
    'frns_sub': '0-40',
    'SVD_Nc': '3',
    'SVD_polyN': '5',
    'sc_factor': 0.995,
    'SVD_fw': '1, 1.5',
    'frns_export': '100-110',
    'export_all': False
}              
              
              
def display_HPLC_data(fn, atsas_path=""):
    """
    This function provides a GUI to access commonly used for interacting with h5sol_HPLC
    
    Parameters
    ----------
    fn : str
        name of the h5 file that contains the in-line HPLC data (scattering and HPLC 
        detectors). The data is expected to be processed already (azimuthal averaging
        and merging). The processed data is expected under the group sample_name/processed.
        The HPLC detector data are expected under the group sample_name/hplc.

    atsas_path : str
        path to ATSAS binary
        
    Returns
    -------
    dt : 
        a h5sol_HPLC instance

    """
    dt = h5sol_HPLC(fn)
    dt.load_d1s()
    dt.set_trans()
    dt.normalize_int()

    if "run_type" not in dt.fh5['/'].attrs.keys():
        dt.fh5['/'].attrs['run_type'] = 'SEC'
        dt.fh5['/'].attrs['instrument'] = "LiX"
        dt.fh5.flush()
    elif dt.fh5['/'].attrs['run_type']!='SEC':
        raise Exception(f"this h5 has been assigned an incompatible run_type: {dt.fh5['/'].attrs['run_type']}")          
              
    # vbox1 1D chromotograms
    vb1Lb = ipywidgets.Label(value="chromatograms:")
    xROI1Tx = ipywidgets.Text(value=HPLC_GUI_par['xroi1'], 
                              description='x-ray ROI1:', 
                              layout=ipywidgets.Layout(width='90%'),
                              style = {'description_width': 'initial'})    
    xROI2Tx = ipywidgets.Text(value=HPLC_GUI_par['xroi2'], 
                              description='x-ray ROI2:', 
                              layout=ipywidgets.Layout(width='90%'),
                              style = {'description_width': 'initial'})    
    xROI3Tx = ipywidgets.Text(value=HPLC_GUI_par['xroi3'], 
                              description='x-ray ROI3:', 
                              layout=ipywidgets.Layout(width='90%'),
                              style = {'description_width': 'initial'})    
    xROIlogCB = ipywidgets.Checkbox(value=HPLC_GUI_par['xroi_log'], 
                                    description='log scale',
                                    layout=ipywidgets.Layout(width='80%'),
                                    style = {'description_width': 'initial'})
    showLC1CB = ipywidgets.Checkbox(value=HPLC_GUI_par['show_LC1'], 
                                    description='LC det 1',
                                    layout=ipywidgets.Layout(width='80%'),
                                    style = {'description_width': 'initial'})
    showLC2CB = ipywidgets.Checkbox(value=HPLC_GUI_par['show_LC2'], 
                                    description='LC det 2',
                                    layout=ipywidgets.Layout(width='80%'),
                                    style = {'description_width': 'initial'})
    btnUpdate = ipywidgets.Button(description='Update plot', 
                                  layout=ipywidgets.Layout(width='35%'))

    vbox1a = ipywidgets.VBox([xROI1Tx, xROI2Tx, xROI3Tx], 
                             layout=ipywidgets.Layout(width='60%'))
    vbox1b = ipywidgets.VBox([xROIlogCB, showLC1CB, showLC2CB], 
                             layout=ipywidgets.Layout(width='40%'))
    vbox1 = ipywidgets.VBox([vb1Lb, ipywidgets.HBox([vbox1a,vbox1b]), btnUpdate],
                            layout=ipywidgets.Layout(width='30%'))
    
    # vbox2, 2D plot parameters
    vb2Lb = ipywidgets.Label(value="2D map:")
    x2dMaplogCB = ipywidgets.Checkbox(value=HPLC_GUI_par['show_2d_log'], 
                                      description='log scale',
                                      layout=ipywidgets.Layout(width='90%'),
                                      style = {'description_width': 'initial'})
    x2dMapSubtractedCB = ipywidgets.Checkbox(value=HPLC_GUI_par['show_2d_sub'], 
                                             description='show subtracted',
                                             layout=ipywidgets.Layout(width='95%'),
                                             style = {'description_width': 'initial'})
    cminTx = ipywidgets.Text(value=HPLC_GUI_par['cmin'], 
                             description='vmin:', 
                             layout=ipywidgets.Layout(width='70%'),
                             style = {'description_width': 'initial'})    
    cmaxTx = ipywidgets.Text(value=HPLC_GUI_par['cmax'], 
                             description='vmax:', 
                             layout=ipywidgets.Layout(width='70%'),
                             style = {'description_width': 'initial'})    
    vbox2 = ipywidgets.VBox([vb2Lb, x2dMapSubtractedCB, x2dMaplogCB, cminTx, cmaxTx], 
                            layout=ipywidgets.Layout(width='15%')) 
    
    # vbox3, backgorund subtraction, export
    vb3Lb = ipywidgets.Label(value="subtraction:")
    subModeDd = ipywidgets.Dropdown(options=['normal', 'SVD'], 
                                    value=HPLC_GUI_par['sub_mode'],
                                    description='mode of subtraction:',
                                    layout=ipywidgets.Layout(width='65%'),
                                    style = {'description_width': 'initial'})
    btnSubtract = ipywidgets.Button(description='subtract', 
                                    layout=ipywidgets.Layout(width='25%'))
    hbox31 = ipywidgets.HBox([subModeDd, btnSubtract])  
    
    frnsSubTx = ipywidgets.Text(value=HPLC_GUI_par['frns_sub'], 
                                description='buffer frames:', 
                                layout=ipywidgets.Layout(width='40%'),
                                style = {'description_width': 'initial'})
    ncTx = ipywidgets.Text(value=HPLC_GUI_par['SVD_Nc'], 
                           description='Nc:', 
                           layout=ipywidgets.Layout(width='15%'),
                           disabled = True,
                           style = {'description_width': 'initial'})
    polyNTx = ipywidgets.Text(value=HPLC_GUI_par['SVD_polyN'], 
                              description='poly N:', 
                              layout=ipywidgets.Layout(width='20%'),
                              disabled = True,
                              style = {'description_width': 'initial'})

    slideScFactor = ipywidgets.FloatSlider(value=HPLC_GUI_par['sc_factor'], 
                                           min=0.8, max=1.2, step=0.0001,
                                           style = {'description_width': 'initial'},
                                           layout=ipywidgets.Layout(width='60%'),
                                           description='Scaling factor:', readout_format='.4f')

    filterWidthTx = ipywidgets.Text(value=HPLC_GUI_par['SVD_fw'], 
                                    description='filter width:', 
                                    layout=ipywidgets.Layout(width='30%'),
                                    disabled = True,
                                    style = {'description_width': 'initial'})

    hbox32a = ipywidgets.HBox([frnsSubTx, ncTx, polyNTx]) 
    hbox32b = ipywidgets.HBox([slideScFactor, filterWidthTx]) 
    
    
    btnExport = ipywidgets.Button(description='Export', 
                                  layout=ipywidgets.Layout(width='20%'))
    frnsExportTx = ipywidgets.Text(value=HPLC_GUI_par['frns_export'], 
                                   description='export frames:', 
                                   layout=ipywidgets.Layout(width='40%'),
                                   style = {'description_width': 'initial'})
    # this is no longer necessary now that RAW can read h5
    #exportAllCB = ipywidgets.Checkbox(value=HPLC_GUI_par['export_all'], 
    #                                  description='export all',
    #                                  layout=ipywidgets.Layout(width='30%'),
    #                                  style = {'description_width': 'initial'})
    
    hbox33 = ipywidgets.HBox([btnExport, frnsExportTx])  #, exportAllCB])      
                        
    vbox3 = ipywidgets.VBox([vb3Lb, hbox31, hbox32a, hbox32b, hbox33], 
                            layout=ipywidgets.Layout(width='45%'))
    
    btnReport = ipywidgets.Button(description='ATSAS report') #, layout=ipywidgets.Layout(width='20%'))
    qSkipTx = ipywidgets.Text(value='0', description='skip:', 
                                layout=ipywidgets.Layout(width='20%'),
                                style = {'description_width': 'initial'})    
    qCutoffTx = ipywidgets.Text(value='0.3', description='q cutoff:', 
                                layout=ipywidgets.Layout(width='25%'),
                                style = {'description_width': 'initial'})    
    outTxt = ipywidgets.Textarea(layout=ipywidgets.Layout(width='60%', height='100%'))
    hbox5 = ipywidgets.HBox([outTxt, 
                             ipywidgets.VBox([btnReport, 
                                              ipywidgets.HBox([qSkipTx, qCutoffTx])]) ])
                        
    box = ipywidgets.HBox([vbox1, vbox2, vbox3])
    display(ipywidgets.VBox([box, hbox5]))
        
    figw = 8
    figh1 = 2
    figh2 = 3
    fig1 = plt.figure(figsize=(figw, figh1+figh2))
    fig2 = plt.figure(figsize=(figw, 3))
    
    def updateDefaults():
        HPLC_GUI_par['xroi1'] = xROI1Tx.value
        HPLC_GUI_par['xroi2'] = xROI2Tx.value
        HPLC_GUI_par['xroi3'] = xROI3Tx.value
        HPLC_GUI_par['xroi_log'] = xROIlogCB.value
        HPLC_GUI_par['show_LC1'] = showLC1CB.value
        HPLC_GUI_par['show_LC2'] = showLC2CB.value
        HPLC_GUI_par['show_2d_log'] = x2dMaplogCB.value
        HPLC_GUI_par['show_2d_sub'] = x2dMapSubtractedCB.value
        HPLC_GUI_par['cmin'] = cminTx.value
        HPLC_GUI_par['cmax'] = cmaxTx.value
        HPLC_GUI_par['sub_mode'] = subModeDd.value
        HPLC_GUI_par['frns_sub'] = frnsSubTx.value
        HPLC_GUI_par['SVD_Nc'] = ncTx.value
        HPLC_GUI_par['SVD_polyN'] = polyNTx.value
        HPLC_GUI_par['sc_factor'] = slideScFactor.value
        HPLC_GUI_par['SVD_fw'] = filterWidthTx.value
        HPLC_GUI_par['frns_export'] = frnsExportTx.value
        #HPLC_GUI_par['export_all'] = exportAllCB.value
    
    def updatePlot(w):
        fig1.clear()
        hfrac = 0.82                
        ht2 = figh1/(figh1+figh2)
        box1 = [0.1, ht2+0.05, hfrac, (0.95-ht2)*hfrac] # left, bottom, width, height
        box2 = [0.1, 0.02, hfrac, ht2*hfrac]
        ax1a = fig1.add_axes(box1)
        ax1b = fig1.add_axes(box2)
        #ax1a = fig1.add_subplot(211)
        #ax1b = fig1.add_subplot(212)
        
        q_ranges = []
        for roiTx in[xROI1Tx, xROI2Tx, xROI3Tx]:
            try:
                q_ranges.append(np.asarray(roiTx.value.split(","), dtype=np.float))
            except:
                pass
            
        dt.plot_data(plot_merged=(not x2dMapSubtractedCB.value), 
                     q_ranges=q_ranges, logROI=xROIlogCB.value,
                     clim=[np.float(cminTx.value), np.float(cmaxTx.value)], 
                     logScale=x2dMaplogCB.value,
                     show_hplc_data=[showLC1CB.value, showLC2CB.value], 
                     ax1=ax1a, ax2=ax1b)
        plt.show(fig1)
        updateDefaults()
    
    def changeSubtractionMode(w):
        HPLC_GUI_par['sub_mode'] = subModeDd.value
        if subModeDd.value=="normal":
            filterWidthTx.disabled = True
            ncTx.disabled = True
            polyNTx.disabled = True
            frnsSubTx.description='buffer frames:'
        else:
            filterWidthTx.disabled = False
            ncTx.disabled = False
            polyNTx.disabled = False
            frnsSubTx.description='excluded frames:'            
    
    #def changeExportMode(w):
    #    HPLC_GUI_par['export_all'] = exportAllCB.value
    #    if exportAllCB.value == True:
    #        frnsExportTx.disabled = True
    #    else:
    #        frnsExportTx.disabled = False
        
    def subtract_buffer(w):
        if subModeDd.value=="normal":
            fig2.clear()
            dt.subtract_buffer(buffer_frame_range=frnsSubTx.value, 
                               sc_factor=slideScFactor.value)
        else: # SVD
            fig2.clear()
            ax2a = fig2.add_subplot(121)
            ax2b = fig2.add_subplot(122)
            filter_width = filterWidthTx.value # e.g. 1 or 0.5, 3 
            try:
                filter_width = float(filter_width)
            except:
                filter_width = json.loads(f"[{filter_width}]")
                if len(filter_width)==0:
                    filter_width = None

            polyN = polyNTx.value
            try:
                polyN = int(polyN)
            except:
                polyN = json.loads(f"[{polyN}]")
                nc = len(polyN)
                ncTx.value = str(nc)
            
            dt.subtract_buffer_SVD(excluded_frames_list = frnsSubTx.value, 
                                  sc_factor = slideScFactor.value,
                                  gaussian_filter_width=filter_width,
                                  Nc = int(ncTx.value), fit_with_polynomial=True, 
                                  poly_order=polyN,
                                  plot_fit=True, ax1=ax2a, ax2=ax2b)
            plt.show(fig2)
            plt.tight_layout()
        
        x2dMapSubtractedCB.value = True
        updatePlot(None)
            
    def export(w):
        updateDefaults()
        fig2.clear()
        if not os.path.isdir("processed/"):
            os.mkdir("processed")
        #if exportAllCB.value:
        #    dt.export_txt(path="processed/")
        #else:
        dt.bin_subtracted_frames(frame_range=frnsExportTx.value,
                                 save_data=True, path="processed/",
                                 fig=fig2, plot_data=True, debug='quiet')
                        
    def report(w):
        updateDefaults()
        fig2.clear()
        if not os.path.isdir("processed/"):
            os.mkdir("processed")
        #if exportAllCB.value:
        #    raise Exception("not implemented")
        d1 = dt.bin_subtracted_frames(frame_range=frnsExportTx.value,
                                      save_data=True, path="processed/",
                                      fig=fig2, plot_data=False, debug='quiet')
        txt = gen_atsas_report(d1, fig=fig2, 
                               skip=int(qSkipTx.value), q_cutoff=float(qCutoffTx.value), 
                               print_results=False, path=atsas_path)
        outTxt.value = txt                
    
    btnUpdate.on_click(updatePlot)
    xROIlogCB.observe(updatePlot)
    showLC1CB.observe(updatePlot)
    showLC2CB.observe(updatePlot)
    x2dMaplogCB.observe(updatePlot)
    x2dMapSubtractedCB.observe(updatePlot)
    
    subModeDd.observe(changeSubtractionMode)
    #exportAllCB.observe(changeExportMode)
    btnSubtract.on_click(subtract_buffer)
    btnReport.on_click(report)
    btnExport.on_click(export)
    
    updatePlot(None)
    
    return dt