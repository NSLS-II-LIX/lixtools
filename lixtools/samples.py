import pandas as pd
import numpy as np
import re

used_at_beamline = False

def check_sample_name(sample_name, ioc_name="pil1M", sub_dir=None, 
                      check_for_duplicate=True, check_dir=False, 
                     ):    
    """
     adapted from 04-sample.py
     when used for data collection at the beamline, 
        root_path = data_path, which is defined as a global variable, f_path = data_path%ioc_name
     when used for data processing and spreadsheet validataion, check name validity only
    """
    
    if len(sample_name)>42:  # file name length limit for Pilatus detectors
        print("Error: the sample name is too long:", len(sample_name))
        return False
    l1 = re.findall('[^:._A-Za-z0-9\-]', sample_name)
    if len(l1)>0:
        print("Error: the file name contain invalid characters: ", l1)
        return False

    if check_for_duplicate and used_at_beamline:
        f_path = data_path
        if sub_dir is not None:
            f_path += ('/'+sub_dir+'/')
        #if DET_replace_data_path:
        #    f_path = data_path.replace(default_data_path_root, substitute_data_path_root)
        #if PilatusFilePlugin.froot == data_file_path.ramdisk:
        #    f_path = data_path.replace(data_file_path.gpfs.value, data_file_path.ramdisk.value)
        if check_dir:
            fl = glob.glob(f_path+sample_name)
        else:
            fl = glob.glob(f_path+sample_name+"_000*")
        if len(fl)>0:
            print(f"Error: name already exists: {sample_name} at {f_path}")
            return False

    return True

def parseSpreadsheet(infilename, sheet_name=0, strFields=[], return_dataframe=False):
    """ dropna removes empty rows
    """
    converter = {col: str for col in strFields} 
    DataFrame = pd.read_excel(infilename, sheet_name=sheet_name, 
                              converters=converter, engine="openpyxl")
    DataFrame.dropna(axis=0, how='all', inplace=True)
    if return_dataframe:
        return DataFrame
    return DataFrame.to_dict()

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

def get_holders_under_config(spreadSheet, configName):
    """
    this function compiles the list of holders under a given config in the storage box
    the spreadsheet must have a Configurations tab, applicable for high-through put solution scattering only
    """
    if configName:
        holders = {}
        d = parseSpreadsheet(spreadSheet, 'Configurations')
        autofillSpreadsheet(d, fields=["Configuration"])
        idx = list(d['Configuration'].keys())
        for i in range(len(idx)):
            if d['Configuration'][idx[i]]==configName:
                holders[d['holderPosition'][idx[i]]] = d['holderName'][idx[i]]

        return holders    

def get_holder_list(spreadSheet):
    d = parseSpreadsheet(spreadSheet, return_dataframe=True)
    hds = d.loc[~d['holderName'].isna(), d.columns.isin(['holderName'])]['holderName']
    return hds.to_list()  
    
def get_blank_holder_dict(spreadSheet):
    d = parseSpreadsheet(spreadSheet, return_dataframe=True)
    hds = d.loc[~d['BlankHolderName'].isna(), d.columns.isin(['holderName', 'BlankHolderName'])]
    return hds.set_index('holderName').to_dict()['BlankHolderName']    
    
def get_empty_holder_dict(spreadSheet):
    d = parseSpreadsheet(spreadSheet, return_dataframe=True)
    hds = d.loc[~d['EmptyHolderName'].isna(), d.columns.isin(['holderName', 'EmptyHolderName'])]
    return hds.set_index('holderName').to_dict()['EmptyHolderName']
                
def parseHolderSpreadsheet(spreadSheet, sheet_name=0, holderName=None,
                check_for_duplicate=False, check_buffer=True, configName=None,
                requiredFields=['sampleName', 'holderName', 'position'],
                optionalFields=['volume', 'exposure', 'bufferName'],
                autofillFields=['holderName', 'volume', 'exposure', 'EmptyHolderName'],
                strFields=['sampleName', 'bufferName', 'holderName'], 
                numFields=['volume', 'position', 'exposure'], 
                min_load_volume=20, maxNsamples=18, sbSameRow=False):
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

    # check sample position validity
    sp = np.asarray(list(d['position'].values()), dtype=int)
    if sp.max()>maxNsamples:
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
        if not check_sample_name(sn, check_for_duplicate=check_for_duplicate):
            raise Exception(f"invalid sample name: {sn} in holder {hn}")
        sdict[hn][pos] = {'sample': sn}
        if str(bn)!='nan':
            sdict[hn][pos]['buffer'] = bn 

    if check_buffer:
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
                if (bpos-pos)%2 and sbSameRow:
                    raise Exception(f"{t['sample']} and its buffer not in the same row in holder {hn}")
                
    if holderName is None:  
        # the correct behavior should be the following:
        # 1. for a holder scheduled to be measured (configName given), there should not be an existing directory
        if configName:
            holders = list(get_holders_under_config(spreadSheet, configName).values())
            for hn in holders:
                if not check_sample_name(hn, check_for_duplicate=used_at_beamline, check_dir=True):
                    raise Exception(f"change holder name: {hn}, already on disk." )
        # 2. for all holders in the spreadsheet, there should not be duplicate names, however, since we are
        #       auto_filling the holderName field, same holder name can be allowed as long as there are no
        #       duplicate sample names. Therefore this is the same as checking for duplicate sample name,
        #       which is already done above
        else:
            holders = []

    # columns in the spreadsheet are dictionarys, not arrays
    idx = list(d['holderName'])  
    hdict = {d['position'][idx[i]]:idx[i] 
             for i in range(len(d['holderName'])) if d['holderName'][idx[i]]==holderName}
        
    samples = {}
    allFields.remove("sampleName")
    allFields.remove("holderName")
    for i in sorted(hdict.keys()):
        sample = {}
        sampleName = d['sampleName'][hdict[i]]
        holderName = d['holderName'][hdict[i]]
        for f in allFields:
            sample[f] = d[f][hdict[i]]
        if "bufferName" in sample.keys():
            if str(sample['bufferName'])=='nan':
                del sample['bufferName']
        samples[sampleName] = sample
            
    return samples

def get_sample_dicts(spreadSheet, holderName, emptySample=None, check_buffer=True):
    """
    check_buffer=True checks whether all necessary buffers are included in the holder, which
        is not necessary for fixed cell measurements
    """
    
    requiredFields = ['sampleName', 'holderName', 'position', 'EmptyHolderName']
    if check_buffer:
        requiredFields += ['bufferName'] 
    ret = {}
    
    samples = parseHolderSpreadsheet(spreadSheet, holderName=holderName, 
                                     requiredFields=requiredFields, check_buffer=check_buffer)
    sdf = pd.DataFrame.from_dict(samples).transpose()
    emptyHolderName = sdf['EmptyHolderName'][0]
    empties = parseHolderSpreadsheet(spreadSheet, holderName=emptyHolderName, check_buffer=check_buffer)
    edf = pd.DataFrame.from_dict(empties).transpose()
    
    sb_dict = {}
    for s in samples.keys():
        if 'bufferName' in samples[s].keys():
            sb_dict[s] = samples[s]['bufferName']   
    ret['buffer'] = sb_dict
        
    se_dict = {}
    if emptySample: # empty cell is one of the samples, applicable when empty cell scattering is reproducible
        if not emptySample in samples.keys():
            raise Exception(f"{emptySample} is not a valid sample for empty cell subtraction ...")
        se_dict = {sn:emptySample for sn in samples.keys() if sn!=emptySample}
    else:
        all_samples = list(set(sb_dict.keys()) | set(sb_dict.values())) 
        for s in samples.keys():
            se_dict[s] = edf.index[edf['position']==samples[s]['position']].values[0]
    ret['empty'] = se_dict

    return ret