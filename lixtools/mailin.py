import pandas as pd
import numpy as np
from itertools import groupby,chain
from io import BytesIO
from os.path import dirname
from barcode import generate,writer
import PIL,openpyxl,subprocess,json
import pylab as plt
import glob,uuid,re,pathlib,os,time
import ipywidgets
from IPython.display import display,clear_output
import webbrowser,qrcode,base64,threading
from collections import Counter,OrderedDict

USE_SHORT_QR_CODE=False

def getUID():
    qstr = str(uuid.uuid4())
    if USE_SHORT_QR_CODE:
        return "-".join(qstr.split("-")[:-1])
    return qstr

def makeQRxls(fn, n=10, use_UID_text=True):
    qdict = {"UIDs" : [getUID() for _ in  range(n)] }
    if use_UID_text:
        qdict['label'] = [uid[0:4] for uid in qdict['UIDs']]
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
                         generate_barcode=True, sh_name=None, check_template=True,
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
    df = xl.parse(sh_name)
    df['Volume (uL)'] = pd.to_numeric(df['Volume (uL)'], errors='coerce')
    
    # template spreadsheet is protected, "lix template"
    if check_template:
        sdict = df.to_dict()['Notes']
        valid = True
        if len(sdict)<99: 
            valid = False
        elif sdict[98]!='lix template':
            valid = False
        if not valid:
            raise Exception("This spreadsheet does not appear to be generated using the LiX template.")
    
    # get all samples
    df1 = df[~df["Sample"].isnull() & df["Stock"].isnull()][["Sample", "Buffer", "Well", "Volume (uL)"]]

    # check redundant sample name, and sample name validity
    all_samples = list(df1['Sample'])
    all_buffers = list(df1[~df1["Buffer"].isnull()]['Buffer'])
    all_sample_wells = list(df1['Well']) 
    for key,group in groupby(all_samples):
        if not check_sample_name(key, check_for_duplicate=False):
            raise Exception(f"invalid sample name: '{key}'")
        if len(list(group))>1:
            raise Exception(f"redundant sample name: {key}")

    # check buffer name, and how many samples use the same buffer for subtraction
    for key, group in groupby(all_buffers):
        if not key in all_samples:
            raise Exception(f"'{key}' is not a sample and therefore not a valid buffer.")
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
