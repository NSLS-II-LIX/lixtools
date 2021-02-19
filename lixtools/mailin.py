import pandas as pd
import numpy as np
from itertools import groupby,chain
from io import BytesIO
from os.path import dirname
from barcode import generate,writer
import PIL,openpyxl,subprocess,json
import pylab as plt
import glob,uuid,re,pathlib,os
import ipywidgets
from IPython.display import display,clear_output
import webbrowser,qrcode,base64

USE_SHORT_QR_CODE=True

def getUID():
    qstr = str(uuid.uuid4())
    if USE_SHORT_QR_CODE:
        return "-".join(qstr.split("-")[:-1])
    return qstr

def makeQRxls(fn, n=10):
    qdict = {"UIDs" : [getUID() for _ in  range(n)] }
    df = pd.DataFrame.from_dict(qdict)
    df.to_excel(fn, index=False)

def make_plate_QR_code(proposal_id, SAF_id, plate_id, path=""):
    """ depends on blabel
        generate a pdf file, with plate outline
    """
    code = [str(proposal_id), str(SAF_id), str(plate_id)]
    if len(code[0])!=6 or len(code[1])!=6:
        raise Exception("Proposal and SAF IDs should each have 6 digits.")
    if len(code[2])!=2:
        raise Exception("Plate IDs should have 2 digits.")
    str_in = '-'.join(code)
    fn = f"plate_{str_in}.html"
    if path!="":
        fn = os.path.join(path, fn)

    lixtools_dir = os.path.dirname(os.path.realpath(__file__))
    template_fn = os.path.join(lixtools_dir, "plate_label_template.html")
    with open(template_fn, "r") as fh:
        txt = fh.read()
    img = qrcode.make(str_in)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    bb = img_byte_arr.getvalue()
    txt = txt.replace("**CODE**", str_in)
    txt = txt.replace("**IMAGE**", base64.b64encode(bb).decode())
    with open(fn, "w") as fh:
        fh.write(txt)
    webbrowser.open(f'file:///{os.path.realpath(fn)}')
    
    return str_in
    
def make_barcode(proposal_id, SAF_id, plate_id, path="", max_width=2.5, max_height=0.4, 
                 module_height=3.5, module_width=0.25, text_distance=0.5, font_size=10):
    """ proposal and SAF numbers are 6-digits 
        plate number can be an arbitrary identifier
    """
    code = [str(proposal_id), str(SAF_id), str(plate_id)]
    if len(code[0])!=6 or len(code[1])!=6:
        raise Exception("Proposal and SAF IDs should each have 6 digits.")
    if len(code[2])!=2:
        raise Exception("Plate IDs should have 2 digits.")
    str_in = '-'.join(code)
    fp = BytesIO()
    options = dict(module_height=module_height, module_width=module_width, 
                   text_distance=text_distance, font_size=font_size)
    generate('code128', str_in, writer=writer.ImageWriter(), output=fp, writer_options=options)
    img = PIL.Image.open(fp)
    dpi = max(img.size[0]/max_width, img.size[1]/max_height)
    pdf_fn = str_in+".pdf"
    if path!="":
        pdf_fn = path+"/"+pdf_fn
    img.save(pdf_fn, resolution=dpi)
    return str_in,img
    
def validate_sample_list(xls_fn, 
                         generate_barcode=True, sh_name=None,
                         proposal_id=None, SAF_id=None, plate_id=None,
                         b_lim = 3,    # number of samples allowed to share the same buffer for subtraction
                         v_margin = 5, # minimal extra volume required in stock wells
                        ):
    """ validate the sample list spreadsheet
        read first sheet, unless sh_name is specified 
        produce the bar code as a pdf file if generate_barcode is True, also rename the sheet name
        otherwise generate the info need for sample mixing/transfer: sample dictionary and list of mixing ops
    """

    msg = []
    
    xl = pd.ExcelFile(xls_fn)
    if sh_name is None:
        sh_name = xl.sheet_names[0]
    msg.append(f"reading data from {sh_name} ...") 
    df = xl.parse(sh_name, dtype={'Volume (uL)': float})    
        
    # get all samples
    df1 = df[~df["Sample"].isnull() & df["Stock"].isnull()][["Sample", "Buffer", "Well", "Volume (uL)"]]
    # check redundant sample name
    all_samples = list(df1['Sample'])
    all_buffers = list(df1[~df1["Buffer"].isnull()]['Buffer'])
    all_sample_wells = list(df1['Well']) 
    for key, group in groupby(all_samples):
        if len(list(group))>1:
            raise Exception(f"redundant sample name: {key}")

    # check how many samples use the same buffer for subtraction
    for key, group in groupby(all_buffers):
        if len(list(group))>b_lim:
            raise Exception(f"{key} is used more than {b_lim} times for buffer subtraction.")

    sdict = df1.set_index("Sample").T.to_dict()
    # check whether every sample has a buffer and in the same row
    for sn in list(set(all_samples)-set(all_buffers)):
        if not isinstance(sdict[sn]["Buffer"], str): # nan
            msg.append(f"Warning: {sn} does not have a corresponding sample/buffer.")
        else:
            swell = sdict[sn]["Well"]
            bwell = sdict[sdict[sn]["Buffer"]]["Well"]
            if swell[0]!=bwell[0]:
                raise Exception(f"sample and buffer not in the same row: {swell} vs {bwell}")
                
    wdict = df[~df["Sample"].isnull()].set_index("Well").T.to_dict()
    wlist = list(wdict.keys())
    # each well either need to have volume specified, or how to generate the sample
    mlist = {}   # source well for each new sample
    slist = {}   # total volume out of source well
    for wn in wlist:
        if isinstance(wdict[wn]["Mixing"], str): 
            tl = {}
            for tt in wdict[wn]['Mixing'].strip().split(","):
                w,v = tt.strip().split(":")
                if w not in wlist:
                    raise Exception(f"source well {w} is empty.")
                if not isinstance(wdict[w]['Stock'], str):
                    raise Exception(f"source well {w} is not designated as a stock well.")
                if w in tl.keys():
                    raise Exception(f"{w} appears more than once in the mixing list for {wn}.")
                tl[w] = float(v)
                if not w in slist.keys():
                    slist[w] = float(v)
                else:
                    slist[w] += float(v)
            mlist[wn] = tl 
        elif np.isnan(wdict[wn]["Volume (uL)"]):
            raise Exception(f"neither volume nor mixing is specified for well {wn}")  

    for wn in slist.keys():
        if slist[wn]+v_margin > wdict[wn]["Volume (uL)"]:
            raise Exception(f"not sufficient sample in well {wn}, need {slist[wn]+v_margin}")
            
    rdict = {}
    for w in all_sample_wells:
        rn = w[0]
        if rn in rdict.keys():
            rdict[rn] += [w]
        else:
            rdict[rn] = [w]

    rlen = np.asarray([len(rdict[rn]) for rn in rdict.keys()])
    if len(rlen[rlen<5])>0:
        msg.append(f"Consider consolidating the rows, more than two rows are half full.")

    if generate_barcode:
        bcode = make_plate_QR_code(proposal_id, SAF_id, plate_id, dirname(xls_fn))
        #bcode,bcode_img = make_barcode(proposal_id, SAF_id, plate_id, dirname(xls_fn))
        #msg.append(f"writing barcode to {bcode}.pdf ...")
        #plt.imshow(bcode_img)
        #display(bcode_img)
        #t=plt.axis(False)
        if sh_name!=bcode:
            msg.append(f"Renaming the sheet name to {bcode} ...")
            ss = openpyxl.load_workbook(xls_fn)
            ss[sh_name].title = bcode
            ss.save(xls_fn)
        return msg
    else:
        bcode = sh_name
        try:
            proposal_id,SAF_id,plate_id = bcode.split('-')
        except:
            raise Exception(f"The bar code (sheet name) does not seem to have the correct format: {bcode}")
        else:
            if len(proposal_id)!=6 or len(SAF_id)!=6 or len(plate_id)!=2:
                raise Exception(f"The bar code (sheet name) does not seem to have the correct format: {bcode}")
        bdict = {"proposal_id": proposal_id, 
                 "SAF_id": SAF_id, 
                 "plate_id": plate_id} 
        print(bdict)
        return bdict,df1.set_index("Well").T.to_dict(),mlist


def process_sample_lists(xls_fns, process_all_tabs=False, b_lim=4):
    """ call validate_sample_list() to get barcode and well dictionary
        ot2_layout should describe the labware present on the deck, with identifying bar/QR codes
    """
    pdict = {}
    bdict0 = None
    if not isinstance(xls_fns, list):
        xls_fns = [xls_fns]
    for fn in xls_fns:
        xl = pd.ExcelFile(fn)
        if process_all_tabs:
            sh_list = xl.sheet_names
        else:
            sh_list = [xl.sheet_names[0]]
        for sh_name in sh_list:
            bdict,wdict,mdict = validate_sample_list(fn, b_lim=b_lim, generate_barcode=False)
            if bdict0 is None:
                bdict0 = bdict
            elif bdict['proposal_id']!=bdict0['proposal_id'] or bdict['proposal_id']!=bdict0['proposal_id']:
                raise Exception("Cannot process plates from differernt proposal/SAF together.")
            pdict[sh_name] = [wdict, mdict]
    
    # get ot2 layout info
    
    # create tranfer list
    transfer_list = []
    slist = []
    for bcode in pdict.keys():
        pname = bcode # labware name for the plate
        plabel = bcode.split("-")[-1]
        row_name = None
        row_count = 0
        wdict,mdict = pdict[bcode]
        for wn in wdict.keys():
            # get labware name for the plate/barcode 
            if row_name!=wn[0]:
                row_name=wn[0]
                row_count += 1
                if row_count%2:
                    # get labware name for the holder
                    hname = f"{plabel}_h{int((row_count+1)/2):02d}"
            col = int(wn[1:])
            hpos = 2*col - (row_count%2) 
            
            if np.isnan(wdict[wn]["Volume (uL)"]):
                vt = 0
            else:
                vt = wdict[wn]["Volume (uL)"]
            # require mixing
            if wn in mdict.keys():
                # mix first
                for ws,v in mdict[wn].items():
                    transfer_list.append([pname, ws, pname, wn, v])
                    vt += v

            # now transfer
            transfer_list.append([pname, wn, hname, hpos, vt])
            
            # add entry into the sample dictionary
            slist.append([hname, hpos, wdict[wn]["Sample"], wdict[wn]["Buffer"], vt])

    return slist,transfer_list,bdict

def read_OT2_layout(plate_slots, holder_slots, msg=None):
    """ 
    the arguments should be a comma-separated list of slot positions on the Opentron deck
    e.g. "1,2"  
    """
    if msg is None:
        ssh_key = str(pathlib.Path.home())+"/.ssh/ot2_ssh_key"
        if not os.path.isfile(ssh_key):
            raise Exception(f"{ssh_key} does not exist!")

        cmd = ["ssh", "-i", ssh_key, "-o", "port=9999",
               "root@localhost", "/var/lib/jupyter/notebooks/check_deck_config.py", 
               "-h", holder_slots, "-p", plate_slots]

        ret = subprocess.run(cmd, capture_output=True)

        if ret.returncode:
            print(ret)
            raise Exception("error executing check_deck_config.py")
        
        msg = ret.stdout.decode()
            
    ldict = json.loads(msg.split("*****")[-1])
    
    return ldict

def generate_measurement_spreadsheet(fn, slist, holders, bdict):
    """ slist is generated by process_sample_lists()
        
        holders is a dictionary of {holder_name: UID}, derived values returned 
        from read_OT2_layout(). this information needs to be saved in the spreadsheet, 
        on the "UIDs" tab, so that the holder name can be found based on the QR 
        code during the measurements
    """
    tslist = np.asarray(slist).T
    sdict = {}
    sdict["holderName"] = tslist[0]
    sdict["position"] = tslist[1]
    sdict["sampleName"] = tslist[2]
    sdict["bufferName"] = ["" if x=="nan" else x for x in tslist[3]]
    sdict["volume"] = tslist[4]
    df1 = pd.DataFrame.from_dict(sdict)

    cdict = {}
    cdict["holderName"] = holders.keys()
    cdict["UID"] = holders.values()
    df2 = pd.DataFrame.from_dict(cdict)

    tname = f"{bdict['proposal_id']}-{bdict['SAF_id']}"
    
    with pd.ExcelWriter(fn) as writer:
        df1.to_excel(writer, index=False, sheet_name=tname)
        df2.to_excel(writer, index=False, sheet_name="UIDs")
            
def generate_docs(ot2_layout, xls_fns, ldict=None,   
                  run_name="test",
                  b_lim=4,
                  plate_type = "corning_96_wellplate_360ul_flat",
                  holder_type = "lix_3x_holder_c",
                  tip_type = "opentrons_96_tiprack_300ul",
                  flow_rate_aspirate = 50, flow_rate_dispense = 50, 
                  bottom_clearance = 1
                 ):
    """ ot2_layout should be a dictionary:
            {"plates" : "1,2",
             "holders" : "7,8",
             "tips" : "9,10"}
    """
    print("Processing sample list(s) ...")
    slist,transfer_list,bdict = process_sample_lists(xls_fns, b_lim=b_lim)
    
    if ldict is None:
        print("Reading bar/QR codes, this might take a while ...")
        ldict = read_OT2_layout(ot2_layout["plates"], ot2_layout["holders"])
    
    print(ldict)
    
    holders = {}
    holder_qr_codes = chain(ldict['holders'].keys())
    print(f"{len(ldict['holders'])} holders are available.")
    for st in slist:
        if not st[0] in holders.keys():
            try:
                holders[st[0]]= next(holder_qr_codes)
            except StopIteration:
                print("Error: Not enough sample holders for transfer.")
                raise
    print(f"{len(holders)} holders are needed.")

    fn = f"{run_name}_protocol.py"
    print(f"Generating protocol ({fn}) ...")
    protocol = ["metadata = {'protocolName': 'sample transfer',\n",
                "            'author': 'LiX',\n",
                "            'description': 'auto-generated',\n",
                "            'apiLevel': '2.3'\n",
                "           }\n", 
                "\n",
                "def run(ctx):\n",]

    for slot in ot2_layout["plates"].split(","):
        protocol.append(f"    lbw{slot} = ctx.load_labware('{plate_type}', '{slot}')\n")
    for slot in ot2_layout["holders"].split(","):
        protocol.append(f"    lbw{slot} = ctx.load_labware('{holder_type}', '{slot}')\n")
    tips = []
    for slot in ot2_layout["tips"].split(","):
        protocol.append(f"    lbw{slot} = ctx.load_labware('{tip_type}', '{slot}')\n")
        tips.append(f"lbw{slot}")
    protocol.append(f"    pipet = ctx.load_instrument('p300_single', 'left', tip_racks=[{','.join(tips)}])\n")
    protocol.append(f"    pipet.well_bottom_clearance.aspirate = {bottom_clearance}\n")
    protocol.append(f"    pipet.flow_rate.aspirate = {flow_rate_aspirate}\n")
    protocol.append(f"    pipet.flow_rate.dispense = {flow_rate_dispense}\n")

    for st in transfer_list:
        src,sw,dest,dw,vol = st
        if dest in holders.keys():
            dest = holders[dest]
        if src in ldict["plates"].keys():
            sname = f"lbw{ldict['plates'][src]['slot']}.well('{sw}')"
        elif src in holders.values():
            sname = f"lbw{ldict['holders'][src]['slot']}.well('{ldict['holders'][src]['holder']}{sw}')"
        else:
            raise Exception(f"Unknown labware encountered: {src}")
        if dest in ldict["plates"].keys():
            dname = f"lbw{ldict['plates'][dest]['slot']}.well('{dw}')"
        elif dest in holders.values():
            dname = f"lbw{ldict['holders'][dest]['slot']}.well('{ldict['holders'][dest]['holder']}{dw}')"
        else:
            raise Exception(f"Unknown labware encountered: {dest}")

        protocol.append(f"    pipet.transfer({vol}, {sname}, {dname})\n")

    fd = open(fn, "w+")
    fd.writelines(protocol)
    fd.close()
    
    fn = f"{run_name}.xlsx"
    print(f"Writing measurement sequence to {fn}.")
    generate_measurement_spreadsheet(fn, slist, holders, bdict)
    
    print("Done.")
    
def generate_docs2(ot2_layout, xls_fns,    
                   run_name="test",
                   b_lim=4,
                   plate_types = ["corning_96_wellplate_360ul_flat", 
                                  "biorad_96_wellplate_200ul_pcr"],
                   holder_types = ["lix_3x_holder_c"],
                   tip_types = ["opentrons_96_tiprack_300ul",
                                "opentrons_96_tiprack_20ul"],
                   pipets = {"left": {"type": "p300_single", "tip_size": "300ul", "maxV": 300}, 
                             "right": {"type": "p20_single", "tip_size": "20ul", "maxV": 20}},
                   flow_rate_aspirate = 0.3, flow_rate_dispense = 0.3  # fraction of the maxV
                  ):
    """ ot2_layout should be a dictionary, slot #: labware type
            {"1" : "lix_3x_holder_c",
             "2" : "corning_96_wellplate_360ul_flat",
             "3" : "opentrons_96_tiprack_300ul"
             "4" : "lix_3x_holder_c",
             "6" : "opentrons_96_tiprack_20ul"}
    """
    print("Processing sample list(s) ...")
    slist,transfer_list,bdict = process_sample_lists(xls_fns, b_lim=b_lim)
    
    h_slots = [k for k,l in ot2_layout.items() if l in holder_types]
    p_slots = [k for k,l in ot2_layout.items() if l in plate_types]
    t_slots = [k for k,l in ot2_layout.items() if l in tip_types]
                    
    print("Reading bar/QR codes, this might take a while ...")
    ldict = read_OT2_layout(",".join(p_slots), ",".join(h_slots))
    print(ldict)
    
    holders = {}
    holder_qr_codes = chain(ldict['holders'].keys())
    print(f"{len(ldict['holders'])} holders are available.")
    for st in slist:
        if not st[0] in holders.keys():
            try:
                holders[st[0]]= next(holder_qr_codes)
            except StopIteration:
                print("Error: Not enough sample holders for transfer.")
                raise
    print(f"{len(holders)} holders are needed.")

    fn = f"{run_name}_protocol.py"
    print(f"Generating protocol ({fn}) ...")
    protocol = ["metadata = {'protocolName': 'sample transfer',\n",
                "            'author': 'LiX',\n",
                "            'description': 'auto-generated',\n",
                "            'apiLevel': '2.3'\n",
                "           }\n", 
                "\n",
                "def run(ctx):\n",]

    for slot in p_slots+h_slots+t_slots:
        protocol.append(f"    lbw{slot} = ctx.load_labware('{ot2_layout[slot]}', '{slot}')\n")

    for k,p in pipets.items():
        tips = ','.join([f"lbw{s}" for s in t_slots if p["tip_size"] in ot2_layout[s]])
        protocol.append(f"    pipet_{k} = ctx.load_instrument('p300_single', 'left', tip_racks=[{tips}])\n")
        protocol.append(f"    pipet_{k}.flow_rate.aspirate = {flow_rate_aspirate*p['maxV']}\n")
        protocol.append(f"    pipet_{k}.flow_rate.dispense = {flow_rate_dispense*p['maxV']}\n")

    # sorted by maxV, low to high
    pvdict = {pipets[pn]["maxV"]:pn for pn in pipets.keys()}
    pvdict = {k:pvdict[k] for k in sorted(pvdict.keys())}
    vlist = list(pvdict.keys())
    def select_pipet(v):
        if (v>vlist).all():
            raise Exception(f"requested transfer volume exceeds tip maximum")
        elif (v<vlist).all():
            p = pvdict[vlist[0]]
        else:
            p = pvdict[vlist[-1]]
        return f"pipet_{p}"         
    
    for st in transfer_list:
        src,sw,dest,dw,vol = st
        if dest in holders.keys():
            dest = holders[dest]
        if src in ldict["plates"].keys():
            sname = f"lbw{ldict['plates'][src]['slot']}.well('{sw}')"
        elif src in holders.values():
            sname = f"lbw{ldict['holders'][src]['slot']}.well('{ldict['holders'][src]['holder']}{sw}')"
        else:
            raise Exception(f"Unknown labware encountered: {src}")
        if dest in ldict["plates"].keys():
            dname = f"lbw{ldict['plates'][dest]['slot']}.well('{dw}')"
        elif dest in holders.values():
            dname = f"lbw{ldict['holders'][dest]['slot']}.well('{ldict['holders'][dest]['holder']}{dw}')"
        else:
            raise Exception(f"Unknown labware encountered: {dest}")
        
        protocol.append(f"    {select_pipet(vol)}.transfer({vol}, {sname}, {dname})\n")

    fd = open(fn, "w+")
    fd.writelines(protocol)
    fd.close()
    
    fn = f"{run_name}.xlsx"
    print(f"Writing measurement sequence to {fn}.")
    generate_measurement_spreadsheet(fn, slist, holders, bdict)
    
    print("Done.")
    
    
def validatePlateSampleListGUI():
    propTx = ipywidgets.Text(value='',
                             layout=ipywidgets.Layout(width='20%'),
                            description='Proposal:')
    safTx = ipywidgets.Text(value='',
                            layout=ipywidgets.Layout(width='20%'), 
                            description='SAF:')
    plateTx = ipywidgets.Text(value='', 
                              layout=ipywidgets.Layout(width='16%'), 
                              description='plate ID:')

    fnFU = ipywidgets.FileUpload(accept='.xlsx', multiple=False, 
                                 description="sample list upload", 
                                 layout=ipywidgets.Layout(width='30%'))
    
    btnValidate = ipywidgets.Button(description='Validate', 
                                    layout=ipywidgets.Layout(width='25%'), 
                                    style = {'description_width': 'initial'})

    outTxt = ipywidgets.Textarea(layout=ipywidgets.Layout(width='55%'))

    hbox1 = ipywidgets.HBox([propTx, safTx, plateTx])                
    hbox2 = ipywidgets.HBox([fnFU, btnValidate])                
    vbox = ipywidgets.VBox([hbox1, hbox2, outTxt])                

    def on_validate_clicked(b):
        flist = list(fnFU.value.keys())
        if len(flist)==0:
            outTxt.value = "upload the sample list spreadsheet first ..."
            return
        try:
            msg = validate_sample_list(flist[0], generate_barcode=True, 
                                 proposal_id=propTx.value, 
                                 SAF_id=safTx.value, 
                                 plate_id=plateTx.value)
            outTxt.value = "\n".join(msg)
        except Exception as e:
            s,r = getattr(e, 'message', str(e)), getattr(e, 'message', repr(e))
            outTxt.value = "Error: "+s

    display(vbox)
    btnValidate.on_click(on_validate_clicked)
    

# adapted from 04-sample.py
def check_sample_name(sample_name, sub_dir=None, 
                      check_for_duplicate=True, check_dir=False, 
                      data_path="./" # global variable in 04-sample.py
                     ):    
    if len(sample_name)>42:  # file name length limit for Pilatus detectors
        print("Error: the sample name is too long:", len(sample_name))
        return False
    l1 = re.findall('[^:._A-Za-z0-9\-]', sample_name)
    if len(l1)>0:
        print("Error: the file name contain invalid characters: ", l1)
        return False

    if check_for_duplicate:
        f_path = data_path
        if sub_dir is not None:
            f_path += ('/'+sub_dir+'/')
        #if DET_replace_data_path:
            #f_path = data_path.replace(default_data_path_root, substitute_data_path_root)
        if PilatusFilePlugin.froot == data_file_path.ramdisk:
            f_path = data_path.replace(data_file_path.gpfs.value, data_file_path.ramdisk.value)
        if check_dir:
            fl = glob.glob(f_path+sample_name)
        else:
            fl = glob.glob(f_path+sample_name+"_000*")
        if len(fl)>0:
            print(f"Error: name already exists: {sample_name} at {f_path}")
            return False

    return True


# adapted from startup_solution.py
def parseSpreadsheet(infilename, sheet_name=0, strFields=[]):
    """ dropna removes empty rows
    """
    converter = {col: str for col in strFields} 
    DataFrame = pd.read_excel(infilename, sheet_name=sheet_name, 
                              converters=converter, engine="openpyxl")
    DataFrame.dropna(axis=0, how='all', inplace=True)
    return DataFrame.to_dict()

def checkHolderSpreadsheet(spreadSheet, sheet_name=0,
                check_for_duplicate=False, configName=None,
                requiredFields=['sampleName', 'holderName', 'position'],
                optionalFields=['volume', 'exposure', 'bufferName'],
                autofillFields=['holderName', 'volume', 'exposure'],
                strFields=['sampleName', 'bufferName', 'holderName'], 
                numFields=['volume', 'position', 'exposure'], 
                min_load_volume=50):
    d = parseSpreadsheet(spreadSheet, sheet_name, strFields)
    tf = set(requiredFields) - set(d.keys())
    if len(tf)>0:
        raise Exception(f"missing fields in spreadsheet: {list(tf)}")
    autofillSpreadsheet(d, fields=autofillFields)
    allFields = list(set(requiredFields+optionalFields).intersection(d.keys()))
    for f in list(set(allFields).intersection(strFields)):
        for e in d[f].values():
            if not isinstance(e, str):
                if not np.isnan(e):
                    raise Exception(f"non-string value in {f}: {e}")
    for f in list(set(allFields).intersection(numFields)):
        for e in d[f].values():
            if not (isinstance(e, int) or isinstance(e, float)):
                raise Exception(f"non-numerical value in {f}: {e}")
            if e<=0 or np.isnan(e):
                raise Exception(f"invalid value in {f}: {e}, positive value required.")
    if 'volume' in allFields:
        if np.min(list(d['volume'].values()))<min_load_volume:
            raise Exception(f"load volume must be greater than {min_load_volume} ul!")

    # max position number is 18
    sp = np.asarray(list(d['position'].values()), dtype=int)
    if sp.max()>18:
        raise Exception(f"invalid sample positionL {sp.max()}.")
    if sp.min()<1:
        raise Exception(f"invalid sample positionL {sp.min()}.")

    sdict = {}
    for (hn,pos,sn,bn) in zip(d['holderName'].values(), 
                              d['position'].values(), 
                              d['sampleName'].values(), 
                              d['bufferName'].values()):
        if not hn in sdict.keys():
            sdict[hn] = {}
        if str(sn)=='nan':
            continue
        if pos in sdict[hn].keys():
            raise Exception(f"duplicate sample position {pos} in {hn}")
        if not check_sample_name(sn, check_for_duplicate=False):
            raise Exception(f"invalid sample name: {sn} in holder {hn}")
        sdict[hn][pos] = {'sample': sn}
        if str(bn)!='nan':
            sdict[hn][pos]['buffer'] = bn 

    for hn,sd in sdict.items():
        plist = list(sd.keys())
        slist = [t['sample'] for t in sd.values()]
        for pos,t in sd.items():
            if slist.count(t['sample'])>1:
                raise Exception(f"duplicate sample name {t['sample']} in {hn}")
            if not 'buffer' in t.keys():
                continue
            if not t['buffer'] in slist:
                raise Exception(f"{t['buffer']} is not a valid buffer in {hn}")
            bpos = plist[slist.index(t['buffer'])]
            if (bpos-pos)%2:
                raise Exception(f"{t['sample']} and its buffer not in the same row in holder {hn}")
                    
    return sdict

def autofillSpreadsheet(d, fields=['holderName', 'volume']):
    """ if the filed in one of the autofill_fileds is empty, duplicate the value from the previous row
    """
    col_names = list(d.keys())
    n_rows = len(d[col_names[0]])
    if n_rows<=1:
        return
    
    for ff in fields:
        if ff not in d.keys():
            #print(f"invalid column name: {ff}")
            continue
        idx = list(d[ff].keys())
        for i in range(n_rows-1):
            if str(d[ff][idx[i+1]])=='nan':
                d[ff][idx[i+1]] = d[ff][idx[i]] 

def validateHolderSpreadsheet(fn, proposal_id, SAF_id):
    # meant to be used by the users to attach to SAF
    # limit to 3 sample holders per spreadsheet
    # validate sample list on the Holders tab and generate UIDs for each holder 
    # beamline prints the QR codes and ship the holders to user
    print("Checking spreadsheet format ...")
    sdict = checkHolderSpreadsheet(fn)
    hlist = list(sdict.keys()) 
    if len(hlist)>3:
        raise Exception(f"Found {len(hlist)} sample holders. Only 3 are allowed.")
    ll = np.asarray([len(h) for h in hlist])
    if (ll>5).any():  # for the purpose of fitting the text on the QR code stciker
        raise Exception(f"Please limit the length of the holder names to 5 characters.")
        
    print("Generating UUIDs ...")
    wb = openpyxl.load_workbook(fn)
    wb[wb.sheetnames[0]].title = f"{proposal_id}-{SAF_id}"
    if "UIDs" in wb.sheetnames:
        del wb["UIDs"]
    ws1 = wb.create_sheet("UIDs")
    ws1.append(["holderName", "UID"])
    for i in range(len(hlist)):
        ws1.append([hlist[i], getUID()])
    wb.save(fn)
    
    print("Done.")
