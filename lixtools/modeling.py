import time,subprocess,os,uuid,glob
from dask.distributed import as_completed
from lixtools.atsas import extract_vals
import numpy as np
import re

def create_uid():
    tt = time.localtime()
    uid = f"{tt.tm_yday:03d}-{tt.tm_hour:02d}:{tt.tm_min:02d}:{tt.tm_sec:02d}-"
    return uid+f"{str(uuid.uuid4())[:8]}"


def run_task(client, cmd, cwd, prerequire=[], shell=False, quiet=False):
    """ run cmd, in cwd
        cmd should be a list (*args), if shell is False
        when wildcards are used, shell should be Ture, and cmd is just a string
        prerequire is a list of futures that must be gathered before the cmd can run
    """
    if not quiet:
        print(f"starting job {cmd} in {cwd}")
    client.gather(prerequire)
    return client.submit(subprocess.run, cmd,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                         shell=shell, check=True, cwd=cwd, key=create_uid())
        
server_atsas_path = "/opt/apps/ATSAS/2.8.5-0/bin"
#server_atsas_path = "/usr/bin"

def getNSD(msg):
    # msg is the output from supcomb 
    # NSD appears to be the smalles value in the final distance column in the output
    # Read file .............................................. : t-00-1.pdb
    # Number of atoms ........................................ : 236
    # Fineness of the structure .............................. : 8.400
    # Read file .............................................. : t-01-1.pdb
    # Number of atoms ........................................ : 236
    # Fineness of the structure .............................. : 8.400
    #
    #                                Initial       Final
    #              Orientation      Distance    Distance
    #             1     1     1       1.3553      1.1500
    #             1     1    -1       1.2461      1.1552
    #             1    -1     1       1.4394      1.1577
    #             1    -1    -1       1.2884      1.1466
    #            -1     1     1       1.8763      1.5808
    #            -1     1    -1       1.9531      1.4054
    #            -1    -1     1       1.8619      1.4467
    #            -1    -1    -1       1.9294      1.4217
    #
    #                   Transformation matrix
    #         0.5193     -0.7225      0.4565     -0.9402
    #         0.0398     -0.5131     -0.8574     -6.5252
    #         0.8537      0.4634     -0.2377     -0.5459
    #         0.0000      0.0000      0.0000      1.0000
    #
    # Wrote file ............................................. : t-01-1r.pdb
    lines = msg.split("\n")
    nsd = -1
    for i in range(len(lines)):
        if lines[i].find("Distance")<0:
            continue
        vals = [float(lines[i+j+1].split(" ")[-1]) for j in range(8)]
        nsd = np.min(vals)
        break
    if nsd<0:
        raise Exception("could not find NSD")
    return nsd
    
def getScore(msg):
    """ msg is the output from denss.align.py
    
        $ denss.align.py -f m01.mrc -ref m02.mrc -o tt
        Selecting best enantiomer(s)...
        Aligning to reference...
        tt.mrc written. Score = 0.381
        
        returning 1-Score to be consistent with NSD (lower value is better)
    """
    txt = msg.strip().split("\n")[-1]
    return 1.-float(txt.split('=')[1])
    
def align_using_denss(fns, n1, n2):
    return ["denss.align.py", 
            "-f", fns[n2], "-ref", fns[n1],  
            "-o", f"aligned_{n1:02d}-{n2:02d}"]

def align_using_supcomb(fns, n1, n2):
    return [os.path.join(server_atsas_path, "supcomb"),
            fns[n1], fns[n2], "-o", f"aligned_{n1:02d}-{n2:02d}.pdb"]

def accept_by_chi2(result, chi2_cutoff, prog_name):
    """ returns True is the chi sq value for the model is no higher than the cutoff
        the chi sq value can be found in the .log file for denss, and in the .fir file for atsas
    """
    if chi2_cutoff<0:
        return True

    for line in reversed(re.split("\n|\r", result.stdout.decode())):
        blks = line.split()
        if prog_name=="denss": 
            if len(blks)>2:  
                chi2 = float(blks[1])
                break
        elif prog_name in ["dammin", "gasbori", "gasborp"] and "Final Chi^2 against raw data" in line:
            chi2 = float(blks[-1])
            break
        #elif "gasbor" in prog_name and "Final Chi against raw data" in line: 
        #    chi2 = float(blks[-1])**2
        #    break
        elif prog_name=="dammif" and "Probability that the model fits the data" in line:
            chi2 = float(blks[-1])
            break
        else:
            chi2 = 0
            
    if chi2<=chi2_cutoff and prog_name!="dammif" or chi2>=chi2_cutoff and prog_name=="dammif":
        print(line, ": accepted")
        return chi2 

    print(line, ": rejected")
    return False
    

def merge_models(client, fns, cwd, prerequire, prog_name="dammif", debug=False, chi2_cutoff=-1):
    """ loop over model_i, model_j:
            supcomb/denss.align model_i model_j
        assemble score table (NSD for supcomb, 1-Score for denss.align)
        discard files with score > Mean + 2*Standard deviation (damaver/damsel manual, also recommended by TG)
        average selected/aligned models 
        post-processing: damfilt damaver.pdb, damstart damaver.pdb   
    """

    if prog_name=="denss":
        score_func = getScore
        align_cmd = align_using_denss
        fext = ".mrc"
        # prefix is the last input, last two digits are the seq number
        # ["denss.py", "-f", f"../{dfn}", *args, "-o", f"{sn}-{i:02d}"]
        i_prefix = -1
    else:
        score_func = getNSD
        align_cmd = align_using_supcomb
        fext = ".pdb"
        # prefix is the last input, last two digits are the seq number
        # [os.path.join(server_atsas_path, args[0]), f"../{dfn}", *args[1:], f"--prefix={sn}-{i:02d}"]
        i_prefix = -1
    
    ns = len(fns)
    if ns<2:
        return
    
    futures = []
    print("comparing models as they become available ...")
    res_list = []
    for future,result in as_completed(prerequire, with_results=True):
        res_no = int(result.args[i_prefix][-2:])
        print(f"model #{res_no:02d} completed ...")
        if accept_by_chi2(result, chi2_cutoff, prog_name):
            res_list.append(res_no)
        n = len(res_list)
        if n>2:
            for i in range(n-1):
                n1 = np.min([res_list[i], res_list[-1]])
                n2 = np.max([res_list[i], res_list[-1]])
                # align #n2 to #n1
                f = run_task(client,
                             cmd=align_cmd(fns, n1, n2), 
                             cwd=cwd, quiet=True)
                futures.append(f)
    
    if len(res_list)==0:
        print("no valid models were found ...")
        return []
    elif len(res_list)==1:
        print(f"the only valid model is #{res_list[0]}")
        return [res_list]
    
    #client.gather(futures)
    scores = np.ones([ns, ns])   # rejected models will have score of 1
    for f in futures:
        try:
            res = f.result()
            mfn = res.args[-1]
            # e.g. aligned_13-14 or aligned_13-14.pdb
            i1,i2 = np.asarray(mfn.strip(".pdb").strip("aligned_").split("-"), dtype=int)
            sc = score_func(res.stdout.decode())
            scores[i1][i2] = sc
            scores[i2][i1] = sc
        except:
            print(res)
            raise
        if debug:
            print(f"score({i1:02d},{i2:02d}) = {sc:.3f}")
    # average score for each structure
    for i in range(ns):
        if i in res_list:
            scores[i][i] = np.average([scores[i,j] for j in range(ns) 
                                       if j!=i and scores[i,j]<1])

    # exclude structures with score>avg(score)+2*std
    dd = scores.diagonal()
    if debug:
        print("averaged scores:", dd)
    cutoff = np.average(dd[dd<1])+np.std(dd[dd<1])*2
    selected = [i for i in range(ns) if dd[i]<cutoff]
    print("structures selected: ", selected)
    if len(selected)==0:
        raise Exception("No models are selected based on similarity.")
    
    print("saving selected models ...")
    model_list = [fns[selected[0]]]
    fns_list = []
    for i in selected[1:]:
        #fn0 = os.path.join(cwd, f"aligned_{selected[0]:02d}-{i:02d}{fext}")
        #fn1 = os.path.join(cwd, f"{fns[i][:-4]}r{fext}")
        fn0 = f"aligned_{selected[0]:02d}-{i:02d}{fext}"
        fn1 = f"{fns[i][:-4]}r{fext}"
        if os.path.isfile(os.path.join(cwd, fn0)):
            fns_list.append([fn0, fn1]) 
            model_list.append(fn0)
        else:
            print(f"Warning: {fn0} does not exist!")
    if debug:
        print("selected models: ", model_list)
        print("cwd = {cwd}")
            
    print("averaging selected models ...")
    if prog_name=="denss":
        client.gather(run_task(client, cmd=["denss.average.py", "-f", *model_list, "-o", "denss"], cwd=cwd))
    else:
        client.gather(run_task(client, cmd=[os.path.join(server_atsas_path, "damaver"), *model_list], cwd=cwd))
        client.gather(run_task(client, cmd=[os.path.join(server_atsas_path, "damfilt"), "damaver.pdb"], cwd=cwd))
        client.gather(run_task(client, cmd=[os.path.join(server_atsas_path, "damstart"), "damaver.pdb"], cwd=cwd))

    print("cleaning up ...")        
    #client.gather([client.submit(os.rename, *fns) for fns in fns_list], cwd=cwd)
    futures = [run_task(client, cmd=['cp', *fns], cwd=cwd) for fns in fns_list]
    client.gather(futures)
    #fn_list = client.submit(glob.glob, 
    #                        os.path.join(cwd, f"aligned_*-*{fext}")).result()
    #client.gather(client.submit(lambda x: [os.remove(x0) for x0 in x], fn_list))
    
    return selected


def model_data(client, fn, rep=20, subdir=None, prog_name="dammif", 
               args=["--mode=SLOW"], debug=False, chi2_cutoff=-1): 
    """ repeat a bunch of dammif/dammin/gasbor/denss runs
        Examples:
            prog_name="dammif", ["--mode=SLOW", "--anisometry=O", "--symmetry=P6"]
            prog_name="denss", args=["-m", "M", "-n", "48", "-ncs", "6", "-ncs_axis", "3"])
        then compare/merge on the results
        prog_name must be one of ["denss", "dammif", "dammin", "gasborp", "gasbori"]
        NOTE: for dammif chi2_cutoff is the lower limit for the model fitness
    """
    if not prog_name in ["denss", "dammif", "dammin", "gasborp", "gasbori"]:
        raise Exception(f"{prog_name} is not a supported modeling program.")
        
    if rep>99:
        rep = 99
    dfn = os.path.basename(fn)            # data file name
    sn = dfn[:-4]                         # sample name, strip off the .out extension from dfn
    if subdir is None:
        subdir = sn
    cwd = f"{os.path.dirname(fn)}/{subdir}"   # data file path
    if not os.path.exists(cwd):
        future = run_task(client, cmd=["mkdir", cwd], cwd="./")
        client.gather(future)
    
    if prog_name=="denss":
        # must install denss first, already in PATH
        # use "M" mode to cover high-q data as recommanded by Tom Grant
        model_cmds = [["denss.py", "-f", f"../{dfn}", *args, 
                       "-o", f"{sn}-{i:02d}"] for i in range(rep)]
    else:
        if prog_name=="dammif":
            lb = "--prefix"
        else:
            lb = "--lo"
        model_cmds = [[os.path.join(server_atsas_path, prog_name), f"../{dfn}", *args,
                       f"{lb}={sn}-{i:02d}"] for i in range(rep)]
    futures = [run_task(client, cmd=cmd, cwd=cwd) for cmd in model_cmds]
    
    if prog_name=="denss":
        slist = merge_models(client, [f"{sn}-{i:02d}.mrc" for i in range(rep)], cwd, futures,
                             prog_name, debug=debug, chi2_cutoff=chi2_cutoff)
        fns = ["denss_avg.mrc"]
    else:
        slist = merge_models(client, [f"{sn}-{i:02d}-1.pdb" for i in range(rep)], cwd, futures,
                             prog_name, debug=debug, chi2_cutoff=chi2_cutoff)    
        fns = ["damaver.pdb", "damfilt.pdb", "damstart.pdb"]
    
    # rename files to clarify data origin
    if len(slist)>1:
        for fn in fns:
            client.gather(client.submit(os.rename, f"{cwd}/{fn}", f"{cwd}/{sn}-{fn}"))

    print("done")
    