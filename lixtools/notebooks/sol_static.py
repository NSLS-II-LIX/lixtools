import ipywidgets
from IPython.display import display,clear_output
import numpy as np
import json,os,h5py,glob
from py4xs.slnxs import trans_mode
from py4xs.slnxs import get_font_size
from py4xs.hdf import h5xs,lsh5,create_linked_files
from lixtools.hdf import h5sol_HT,h5sol_HPLC
from lixtools.atsas import gen_atsas_report
import pylab as plt
from scipy import interpolate,integrate

from lixtools.hdf.sol import h5sol_fc

# these are from ipywidget docs
import asyncio
from time import time
from threading import Timer

def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()
        return debounced
    return decorator

def throttle(wait):
    """ Decorator that prevents a function from being called
        more than once every wait period. """
    def decorator(fn):
        time_of_last_call = 0
        scheduled, timer = False, None
        new_args, new_kwargs = None, None
        def throttled(*args, **kwargs):
            nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer
            def call_it():
                nonlocal new_args, new_kwargs, time_of_last_call, scheduled, timer
                time_of_last_call = time()
                fn(*new_args, **new_kwargs)
                scheduled = False
            time_since_last_call = time() - time_of_last_call
            new_args, new_kwargs = args, kwargs
            if not scheduled:
                scheduled = True
                new_wait = max(0, wait - time_since_last_call)
                timer = Timer(new_wait, call_it)
                timer.start()
        return throttled
    return decorator

class solFCgui:
    
    btUpdate = ipywidgets.Button(description='update')
    ddFileList = ipywidgets.Dropdown(description='h5 files:', layout=ipywidgets.Layout(width='90%'))
    slideScFactor = ipywidgets.FloatSlider(value=1.0, min=0.8, max=1.2, step=0.0001,
                                           style = {'description_width': 'initial'},
                                           description='Scaling factor:', readout_format='.4f',
                                           layout=ipywidgets.Layout(width='90%'))
    ddDatakeyList = ipywidgets.Dropdown(description='datakey:')
    vbox1 = ipywidgets.VBox([btUpdate, ddFileList, ddDatakeyList, slideScFactor], layout=ipywidgets.Layout(width='30%'))

    label2 = ipywidgets.Label(value='show data:')
    rbShowData = {}
    rbShowData['sample'] = ipywidgets.RadioButtons(options=['sample'], layout=ipywidgets.Layout(width='90%'))
    rbShowData['blank'] = ipywidgets.RadioButtons(options=['blank'], layout=ipywidgets.Layout(width='90%'), index=None)
    rbShowData['empty'] = ipywidgets.RadioButtons(options=['empty'], layout=ipywidgets.Layout(width='90%'), index=None)
    vboxSDrb = ipywidgets.VBox([rbShowData['sample'], rbShowData['blank'], rbShowData['empty']], 
                               layout=ipywidgets.Layout(width='40%'))

    samples = []
    ddSampleList = ipywidgets.Dropdown(options=samples, layout=ipywidgets.Layout(width='90%'))
    txBlank = ipywidgets.Text(layout=ipywidgets.Layout(width='90%'), disabled=True)
    txEmpty = ipywidgets.Text(layout=ipywidgets.Layout(width='90%'), disabled=True)
    vboxSDnm = ipywidgets.VBox([ddSampleList, txBlank, txEmpty], layout=ipywidgets.Layout(width='60%'))    

    hboxSD = ipywidgets.HBox([vboxSDrb, vboxSDnm], layout=ipywidgets.Layout(align_items='stretch'))   
    vbox2 = ipywidgets.VBox([label2, hboxSD], layout=ipywidgets.Layout(layout=ipywidgets.Layout(width='40%')))

    smAverage = ipywidgets.SelectMultiple(description="frames:")
    btReprocess = ipywidgets.Button(description='re-process')
    vboxAvg = ipywidgets.VBox([smAverage, btReprocess])

    box = ipywidgets.HBox([vbox1, vbox2, vboxAvg])
    #outTxt = ipywidgets.Textarea(layout=ipywidgets.Layout(width='80%', height='50px'))
    #box = ipywidgets.VBox([hbox1, outTxt])

    dt = None
    prev_fn = ""
    prev_sn = None
    prev_scf = 1
    
    def __init__(self, atsas_path=""):
        self.atsas_path = atsas_path
        if not os.path.exists("processed/"):
            os.mkdir("processed")
    
    def observe(self):
        self.ddSampleList.observe(self.onChangeSample)
        self.ddDatakeyList.observe(self.onUpdatePlot)
        self.slideScFactor.observe(self.onScfacrtorChanged)

    def unobserve(self):
        self.ddSampleList.unobserve(self.onChangeSample)
        self.ddDatakeyList.unobserve(self.onUpdatePlot)
        self.slideScFactor.unobserve(self.onScfacrtorChanged)
    
    def run(self, read_only=False):
        self.read_only = read_only
        display(self.box)

        fig1 = plt.figure(figsize=(9, 6))
        self.ax = plt.gca()

        plt.ion()
        self.btUpdate.on_click(self.updateFileList)
        self.btReprocess.on_click(self.reprocess)
        self.rbShowData['sample'].observe(self.onShowDataChanged)
        self.rbShowData['blank'].observe(self.onShowDataChanged)
        self.rbShowData['empty'].observe(self.onShowDataChanged)        
        self.ddFileList.observe(self.onChangeDataFile)
        self.observe()
        
    def onShowDataChanged(self, w):
        if w.name!='index' or w.new is None:
            return
        rblist = ['sample', 'blank', 'empty']
        rblist.remove(w.owner.value)
        for rbn in rblist:
            self.rbShowData[rbn].index = None
        self.onUpdatePlot(None)

    def onChangeDataFile(self, w):
        fn = self.ddFileList.value
        #print(f"in onChangeDataFile, {self.prev_fn} -> {fn}")
        if fn==self.prev_fn or fn is None:
            return
        
        # make sure that the data is the right type, check buffer_list and empty_list??
        self.dt = h5sol_fc(fn)
        self.dt.load_d1s()
        self.samples = sorted(self.dt.buffer_list.keys())
        self.ddSampleList.options = self.samples
        self.ddSampleList.value = self.samples[0]
        self.prev_fn = fn
        
    def updateFileList(self, w):
        flist = []      
        for fn in glob.glob("*.h5"):
            with h5py.File(fn, "r") as fh5:
                md = set(['instrument', 'run_type'])
                if not md.issubset(fh5.attrs):
                    continue
                if fh5.attrs['instrument']=='LiX' and fh5.attrs['run_type']=='fixed cell':
                    flist.append(fn)
        
        self.ddFileList.options = flist
        if len(flist)>0:
            self.ddFileList.value = flist[0]
        else:
            self.ddFileList.value = None
    
    def onChangeSample(self, w):
        if self.dt is None:
            return

        sn = self.ddSampleList.value
        if sn is None or sn==self.prev_sn:
            return
        
        self.unobserve()
        self.txBlank.value = self.dt.buffer_list[sn][0]
        self.txEmpty.value = self.dt.empty_list[sn][0]
        self.slideScFactor.value = self.dt.attrs[sn]['sc_factor']
        self.prev_scf = self.dt.attrs[sn]['sc_factor']
        self.observe()
                    
        self.prev_sn = sn
        self.onUpdatePlot(None)
    
    def onUpdatePlot(self, w):
        if self.dt is None:
            return
        
        self.unobserve()
        self.ax.clear()
        
        if w!='update only':
            all_ks = ['averaged', 'empty_subtracted', 'subtracted']
            self.sn = self.ddSampleList.value
            self.bn = self.dt.buffer_list[self.sn][0]
            self.en = self.dt.empty_list[self.sn][0]
            if self.rbShowData['sample'].index is not None:
                ks = set(self.dt.d1s[self.sn].keys())
                ks = list(ks & set(all_ks))
                self.sns = self.sn
                self.dataShown = 'sample'
            elif self.rbShowData['blank'].index is not None:    
                ks = set(self.dt.d1s[self.bn].keys())
                ks = list(ks & set(all_ks))
                self.sns = self.bn
                self.dataShown = 'blank'
            elif self.rbShowData['empty'].index is not None:
                ks = ['averaged']
                self.sns = self.en
                self.dataShown = 'empty'
            else:
                raise Exception("none of the data type is selected ...")

            self.dk = self.ddDatakeyList.value
            if self.ddDatakeyList.options!=ks:
                if not self.dk in ks:
                    self.dk = ks[0]
                self.ddDatakeyList.options = ks
                self.ddDatakeyList.value = self.dk

            self.slideScFactor.disabled = not (self.dk=='subtracted')
            self.btReprocess.disabled = not (self.dk=='averaged')

        if self.dk=='averaged':
            self.nfr = len(self.dt.attrs[self.sns]['selected'])
            self.frames = [f"frame #{i}" for i in range(self.nfr)]
            self.smAverage.options = self.frames
            sel = [self.frames[i] for i in range(self.nfr) if self.dt.attrs[self.sns]['selected'][i]]
            self.smAverage.value = sel
            self.dt.plot_d1s(self.sns, show_subtracted=False, offset=0.7, ax=self.ax)
        else:
            self.dt.plot_d1s(self.sns, show_subtracted=self.dk, ax=self.ax)
            self.smAverage.options = []

        self.observe()
            
    def onScfacrtorChanged(self, w):
        scf = self.slideScFactor.value
        if np.fabs(scf-self.prev_scf)<0.001:
            return
        # thiis should only happen when the sample/subtracted data are displayed
        sn = self.ddSampleList.value
        self.dt.subtract_buffer(samples=sn, sc_factor=scf, input_grp='empty_subtracted')
        self.prev_scf = scf
        self.onUpdatePlot(None)
    
    def reprocess(self, w):
        """ update frame selection for averaging, also update all other data that are affected
            if empty data is shown, update all effected blank and sample
            if blank data is shown, update all affected sample
            if sample data is shown, update the sample itself only
        """
        
        selected = [(self.frames[i] in self.smAverage.value) for i in range(self.nfr)]
        self.dt.average_d1s(samples=self.sns, selection=selected)
        self.dt.subtract_empty()
        self.dt.subtract_buffer(sc_factor='auto')  # keep current value
        
        self.onUpdatePlot("update only")
            

class solHTgui:
    
    dataFileSel = ipywidgets.Dropdown(description='select data file:', layout=ipywidgets.Layout(width='45%'))
    btUpdateFlist = ipywidgets.Button(description='Update file list', layout=ipywidgets.Layout(width='25%'))
    hbox = ipywidgets.HBox([dataFileSel, btUpdateFlist])    
    
    samples = []
    ddSample = ipywidgets.Dropdown(options=samples, description='Sample:')
    sampleLabels = []
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
    
    box = ipywidgets.VBox([hbox, hbox1, hbox5])
    dt = None
    prev_fn = ""
    prev_sn = None
    
    def __init__(self, atsas_path=""):
        self.atsas_path = atsas_path
        if not os.path.exists("processed/"):
            os.mkdir("processed")
    
    def getFileList(self):
        flist = []
        for fn in glob.glob("*.h5"):
            with h5py.File(fn, "r") as fh5:
                if not set(['detectors', 'instrument']).issubset(list(fh5.attrs)):
                    continue
                if fh5.attrs['run_type']=='static':
                    flist.append(fn)
        return flist
                
    def onChangeDataFile(self, w):
        fn = self.dataFileSel.value
        if fn==self.prev_fn or fn is None:
            return
        if self.dt is not None:
            del self.dt

        self.outTxt.value = f"loadfing file: {fn}"
        if self.dt:
            self.dt.fh5.close()
            del self.dt
        self.dt = h5sol_HT(fn, read_only=self.read_only)
        self.dt.load_d1s()
        self.dt.subtract_buffer(sc_factor=-1, debug='quiet')
    
        with h5py.File(self.dt.fn, "r") as self.dt.fh5:
            if "run_type" not in self.dt.fh5.attrs.keys():
                self.dt.set_h5_attr('/', 'run_type', 'static')
                self.dt.set_h5_attr('/', 'instrument', 'LiX')
            elif self.dt.fh5.attrs['run_type']!='static':
                raise Exception(f"this h5 has been assigned an incompatible run_type: {self.dt.fh5.attrs['run_type']}")

        self.prev_fn = fn
        self.pref_sn = None
        self.ddSample.unobserve(self.onChangeSample)
        self.subtractCB.unobserve(self.onShowSubChanged)

        self.subtractCB.value = False
        self.ddSample.options = self.dt.samples
        self.ddSample.value = self.dt.samples[0]
        self.sampleLabels = [f"frame #{i}" for i in range(len(self.dt.attrs[self.dt.samples[0]]['selected']))]
        self.smAverage.options = self.sampleLabels

        self.ddSample.observe(self.onChangeSample)
        self.subtractCB.observe(self.onShowSubChanged)

        self.onChangeSample(self.dt.samples[0])
        #self.onUpdatePlot(None)
            
    def run(self, read_only=False):
        self.read_only = read_only
        display(self.box)

        fig1 = plt.figure(figsize=(7, 4))
        # rect = l, b, w, h
        self.ax1 = fig1.add_axes([0.1, 0.15, 0.5, 0.78])
        self.ax2 = fig1.add_axes([0.72, 0.61, 0.26, 0.32])
        self.ax3 = fig1.add_axes([0.72, 0.15, 0.26, 0.32])

        self.axr = []
        fig2 = plt.figure(figsize=(7,2.5))
        self.axr.append(fig2.add_axes([0.09, 0.25, 0.25, 0.6])) 
        self.axr.append(fig2.add_axes([0.41, 0.25, 0.25, 0.6])) 
        self.axr.append(fig2.add_axes([0.73, 0.25, 0.25, 0.6])) 
        self.axr.append(self.axr[0].twiny())
        plt.ion()
        
        self.dataFileSel.observe(self.onChangeDataFile)
        self.onChangeSample(None)
        self.btnUpdate.on_click(self.onUpdatePlot)
        self.subtractCB.observe(self.onShowSubChanged)
        self.slideScFactor.observe(self.onUpdatePlot)
        self.ddSample.observe(self.onChangeSample)
        self.btnExport.on_click(self.onExport)
        self.btnReport.on_click(self.onReport)
        self.btUpdateFlist.on_click(self.updateFileList)
        self.updateFileList(None)
        
    
    def updateFileList(self, w):
        flist = self.getFileList()
        self.dataFileSel.options = flist
        if len(flist)>0:
            self.dataFileSel.value = flist[0]
        else:
            self.dataFileSel.value = None
    
    def onChangeSample(self, w):
        if self.dt is None:
            return

        #if self.prev_sn is not None:
        #    self.dt.export_d1s(self.prev_sn, path="processed/", save_subtracted=self.exportSubtractedCB.value)
        sn = self.ddSample.value
        if sn is None:
            return
        
        sel = [self.sampleLabels[i] for i in range(len(self.sampleLabels)) 
               if self.dt.attrs[sn]['selected'][i]]
        self.smAverage.value = sel    
        isSample = ('sc_factor' in self.dt.attrs[sn].keys())
        for a in self.axr:
            a.clear()
        self.outTxt.value = ""

        if isSample:
            self.subtractCB.disabled = False
            self.slideScFactor.value = self.dt.attrs[sn]['sc_factor']
            self.exportSubtractedCB.disabled = False
            if self.subtractCB.value:
                self.btnReport.disabled = False
        else:
            self.subtractCB.value = False
            self.subtractCB.disabled = True
            self.slideScFactor.disabled = True
            self.exportSubtractedCB.value = False
            self.exportSubtractedCB.disabled = True
            self.btnReport.disabled = True
            
        self.prev_sn = sn
        self.onUpdatePlot(None)
    
    def onReport(self, w):
        #try:
        txt = gen_atsas_report(self.dt.d1s[self.ddSample.value]["subtracted"], ax=self.axr, 
                               sn=self.ddSample.value,
                               skip=int(self.qSkipTx.value), q_cutoff=float(self.qCutoffTx.value), 
                               print_results=False, path=self.atsas_path)
        self.outTxt.value = txt
        #except:
        #    outTxt.value = "unable to run ATSAS ..."
    
    def onUpdatePlot(self, w):
        if self.dt is None:
            return
        sn = self.ddSample.value
        re_calc = False
        show_sub = self.subtractCB.value
        sc_factor = self.slideScFactor.value
        sel = [(self.sampleLabels[i] in self.smAverage.value) for i in range(len(self.sampleLabels))]
        isSample = ('sc_factor' in self.dt.attrs[sn].keys())
        if w is not None:
            if np.any(sel != self.dt.attrs[sn]['selected']):
                self.dt.average_d1s(sn, selection=sel, debug=False)
                if isSample:
                    re_calc = True
            if isSample:
                if sc_factor!=self.dt.attrs[sn]['sc_factor']:
                    re_calc = True
            if re_calc:
                self.dt.subtract_buffer(sn, sc_factor=sc_factor, debug='quiet')
                re_calc = False
        self.ax1.clear()
        self.dt.plot_d1s(sn, ax=self.ax1, show_subtracted=show_sub)
        self.ax2.clear()
        self.ax3.clear()
        if isSample and show_sub:
            d1 = self.dt.d1s[sn]['subtracted']
            ym = np.max(d1.data[d1.qgrid>0.5])
            qm = d1.qgrid[d1.data>0][-1]
            self.ax2.semilogy(d1.qgrid, d1.data)
            #ax2.errorbar(d1.qgrid, d1.data, d1.err)
            self.ax2.set_xlim(left=0.5, right=qm)
            self.ax2.set_ylim(top=ym*1.1)
            self.ax2.yaxis.set_major_formatter(plt.NullFormatter())
            qs = float(self.guinierQsTx.value)
            i0,rg,_ = self.dt.d1s[sn]['subtracted'].plot_Guinier(ax=self.ax3, qs=qs, fontsize=0)
            self.ax3.yaxis.set_major_formatter(plt.NullFormatter())
            self.guinierRgTx.value = ("%.2f" % rg)
            #print(f"I0={i0}, Rg={.2f:rg}")
            #plt.tight_layout()
            self.ax2.set_title("buf subtraction")
            self.ax3.set_title("Guinier")
    
    def onShowSubChanged(self, w):
        show_sub = self.subtractCB.value
        if show_sub:
            self.slideScFactor.disabled = False
            self.smAverage.disabled = True
            self.btnReport.disabled = False
        else:
            self.slideScFactor.disabled = True
            self.smAverage.disabled = False
            self.btnReport.disabled = True
        self.onUpdatePlot(None)
    
    def onExport(self, w):
        sn = self.ddSample.value
        self.dt.export_d1s(samples=sn, path="processed/", save_subtracted=self.exportSubtractedCB.value)
        self.dt.update_h5()
        
    
def display_solHT_data(fn, atsas_path="", read_only=False):
    """ atsas_path for windows might be c:\atsas\bin
    """
    dt = h5sol_HT(fn, read_only=read_only)
    dt.load_d1s()
    dt.subtract_buffer(sc_factor=-1, debug='quiet')
    if not os.path.exists("processed/"):
        os.mkdir("processed")
    
    with h5py.File(dt.fn, "r") as dt.fh5:
        if "run_type" not in dt.fh5.attrs.keys():
            dt.set_h5_attr('/', 'run_type', 'static')
            dt.set_h5_attr('/', 'instrument', 'LiX')
        elif dt.fh5.attrs['run_type']!='static':
            raise Exception(f"this h5 has been assigned an incompatible run_type: {dt.fh5.attrs['run_type']}")
    
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
        dt.plot_d1s(sn, ax=ax1, show_subtracted=show_sub)
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
            qs = float(guinierQsTx.value)
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
        
    plt.ion()
    onChangeSample(None)
    btnUpdate.on_click(onUpdatePlot)
    subtractCB.observe(onShowSubChanged)
    slideScFactor.observe(onUpdatePlot)
    ddSample.observe(onChangeSample)
    btnExport.on_click(onExport)
    btnReport.on_click(onReport)
    
    return dt

