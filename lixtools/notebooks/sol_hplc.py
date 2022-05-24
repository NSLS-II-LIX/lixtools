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
    'export_all': False,
    'peak_guess': '125, 140',
    'half_width': '15, 13',
    'guinier_q': '[0.08, 0.11], [0.15, 0.15]',
    'grad_thresh': '100.0, 10.0',
    'opt_method_step1': 'dogbox',
    'opt_method_step2': 'trf',
}              
              
              
def display_HPLC_data(fn, atsas_path="", transField="em2_sum_all_mean_value",
                      read_only=False, transMode=trans_mode.from_waxs):
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
    dt = h5sol_HPLC(fn, transField=transField, read_only=read_only)
    dt.load_d1s()
    dt.set_trans(transMode, trigger="sol", dt0=0.05)  # in case external trans values are needed
    dt.normalize_int()

    if "run_type" not in dt.fh5['/'].attrs.keys():
        dt.enable_write(True)
        dt.fh5['/'].attrs['run_type'] = 'SEC'
        dt.fh5['/'].attrs['instrument'] = "LiX"
        dt.fh5.flush()
        dt.enable_write(False)
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
                                 fig=fig2, plot_data=True, debug='quiet', txtWidget=outTxt)
                        
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
    
    plt.ion()
    updatePlot(None)
    
    return dt