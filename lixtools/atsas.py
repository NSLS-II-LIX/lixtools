import copy,subprocess,os,tempfile,re
import uuid,time,glob,lixtools
import numpy as np
from io import StringIO
import pylab as plt
from dask.distributed import as_completed

def run(cmd, path="", ignoreErrors=True, returnError=False, debug=False):
    """ cmd should be a list, e.g. ["ls", "-lh"]
        path is for the cmd, not the same as cwd
    """
    cmd[0] = path+cmd[0]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if debug:
        print(out.decode(), err.decode())
    if len(err)>0 and not ignoreErrors:
        print(err.decode())
        raise Exception(err.decode())
    if returnError:
        return out.decode(),err.decode()
    else:
        return out.decode()
    
def extract_vals(txt, dtype=float, strip=None, debug=False):
    if strip is not None:
        txt = txt.replace(strip, " ")
    sl = txt.split(" ")
    ret = []
    for ss in sl:
        try:
            val = dtype(ss)
        except:
            pass
        else:
            ret.append(val)
    return ret
    
def atsas_create_temp_file(fn, d1s, skip=0, q_cutoff=0.6):
    idx = (d1s.qgrid<=q_cutoff)
    np.savetxt(fn, np.vstack([d1s.qgrid[idx][skip:], d1s.data[idx][skip:], d1s.err[idx][skip:]]).T)
    
def atsas_autorg(fn, debug=False, path=""):
    ret = run(["autorg", fn], path).split('\n')
    #rg,drg = extract_vals(ret[0], "+/-", debug=debug)
    #i0,di0 = extract_vals(ret[1], "+/-", debug=debug)
    #n1,n2 = extract_vals(ret[2], " to ", debug=debug, dtype=int)
    #qual = extract_vals(ret[3], "%", debug=debug)
    try:
        rg,drg = extract_vals(ret[0])
        i0,di0 = extract_vals(ret[1])
        n1,n2 = extract_vals(ret[2], dtype=int)
        qual = extract_vals(ret[3], strip="%")[0]
    except:
        print("Unable to run autorg ...")
        rg,drg = 0,0
        i0,di0 = 0,0
        n1,n2 = 0,-1
        qual = 0
    
    return {"Rg": rg, "Rg err": drg, 
            "I0": i0, "I0 err": di0,
            "fit range": [n1,n2],
            "quality": qual}

def atsas_datgnom(fn, rg, first, last=None, fn_out=None, path=""):
    """ 
    """
    if fn_out is None:
        fn_out = fn.split('.')[0]+'.out'    
    options = ["-r", str(rg), "-o", fn_out, "--first", str(first)]
    if last is not None:
        options += ["--last", str(last)]
    
    # datgnom vs datgnom4, slightly different input parameters
    ret = run(["datgnom", *options, fn], path).split("\n")
    try:
        if len(ret)>1:
            # example stdout results:
            #    dmax:   50.170000000000002       Total:  0.90463013315175844     
            #    Guinier:   15.332727107179052       Gnom:   15.332431498444064   
            dmax,qual = extract_vals(ret[0])
            rgg,rgp = extract_vals(ret[1])
        else: 
            # newer version of datgnom no longer reports Dmax/Rg on stdout,
            # the .out file format is also different
            # the Rg/Dmax values used to be located at the end of the file:
            #    Total  estimate : 0.944  which is  AN EXCELLENT  solution
            #    Reciprocal space: Rg =   14.75     , I(0) =   0.5740E+01
            #    Real space: Rg =   14.74 +- 0.092  I(0) =   0.5740E+01 +-  0.2274E-01
            # in the more recent files, this info in embedded in the header
            #    Total Estimate:                      0.9546 (a EXCELLENT solution)
            #    Reciprocal space Rg:             0.1471E+02
            #    Reciprocal space I(0):           0.5733E+01
            #    Real space range:                    0.0000 to      45.9000
            #    Real space Rg:                   0.1470E+02 +-   0.1148E+00
            #    Real space I(0):                 0.5733E+01 +-   0.2619E-01
            qual = extract_vals(run(["grep", "Total Estimate", fn_out], path))[0]
            dmax = extract_vals(run(["grep", "Real space range", fn_out], path))[1]
            rgg = extract_vals(run(["grep", "Reciprocal space Rg", fn_out], path))[0]
            rgp = extract_vals(run(["grep", "Real space Rg", fn_out], path))[0]
    except:
        qual = 0
        dmax = 100
        rgg = 0
        rgp = 0
    
    return {"Dmax": dmax, "quality": qual, 
            "Rg (q)": rgg, "Rg (r)": rgp}

def read_arr_from_strings(lines, cols=[0,1,2]):
    """ assuming that any none numerical values will be ignored
        data are in multiple columns
        some columns may be missing values at the top
        for P(r), cols=[0,1,2]
        for I_fit(q), cols=[0,-1]
    """
    ret = []
    for buf in lines:
        if len(buf)<len(cols):  # empty line
            continue        
        tb = np.genfromtxt(StringIO(buf))
        if np.isnan(tb).any():   # mixed text and numbersS          J EXP       ERROR       J REG       I REG
            continue
        ret.append([tb[i] for i in cols])
    return np.asarray(ret).T

def read_gnom_out_file(fn, plot_pr=False, ax=None):
    ff = open(fn, "r")
    tt = ff.read()
    ff.close()
    
    hdr,t1 = tt.split("####      Experimental Data and Fit                     ####")
    #hdr,t1 = tt.split("S          J EXP       ERROR       J REG       I REG")
    iq, pr = t1.split("####      Real Space Data                               ####")
    #iq, pr = t1.split("Distance distribution  function of particle")
    dq, di = read_arr_from_strings(iq.rstrip().split("\n"), cols=[0,-1])
    dr, dpr, dpre = read_arr_from_strings(pr.rstrip().split("\n"), cols=[0,1,2])
    
    if plot_pr:
        if ax is None:
            plt.figure()
            ax = plt.gca()
        ax.errorbar(dr, dpr, dpre)
    
    return hdr.rstrip(),dq,di,dr,dpr,dpre

# ALMERGE   Automatically merges data collected from two different concentrations or 
#           extrapolates it to infinite dilution assuming moderate particle interactions.
#

""" version-dependent output 

$ datporod --version
datporod, ATSAS 2.8.5 (r11116)
Copyright (c) ATSAS Team, EMBL, Hamburg Outstation 2009-2018

$ datporod --help
Usage: datporod [OPTIONS] [FILE(S)]

Estimation of Porod volume. Output values are Rg, I0, estimated Volume and file name.

Known Arguments:
  FILE                       Data file

Known Options:
      --i0=<VALUE>           Forward scattering intensity
      --rg=<VALUE>           Experimental Radius of Gyration
      --first=<N>            index of the first point to be used (default: 1)
      --last=<N>             index of the last point to be used (default: s*Rg ~ 7)
  -h, --help                 Print usage information and exit
  -v, --version              Print version information and exit

Mandatory arguments to long options are mandatory for short options too.


$ datporod --version
datporod, ATSAS 3.0.1 (r12314)
Copyright (c) ATSAS Team, EMBL, Hamburg Outstation 2009-2020

$ datporod --help
Usage: datporod [OPTIONS] [FILE(S)]

Molecular weight from Porod Volume 
Output: smax (A^-1), Volume (A^3), file name

Known Arguments:
  FILE                       Data file

Known Options:
      --i0=<VALUE>           Forward scattering intensity
      --rg=<VALUE>           Experimental Radius of Gyration
      --first=<N>            index of the first point to be used (default: 1)
      --last=<N>             index of the last point to be used (default: s*Rg ~ 7)
  -h, --help                 Print usage information and exit
  -v, --version              Print version information and exit

Mandatory arguments to long options are mandatory for short options too.



datmow --version
datmow, ATSAS 2.8.5 (r11116)
Copyright (c) ATSAS Team, EMBL, Hamburg Outstation 2014-2018

$ datmow --help
Usage: datmow [OPTIONS] [FILE(S)]

Output: Q', V' (apparent Volume), V (Volume, A^3), MW (Da), file name

Known Arguments:
  FILE                       Data file

Known Options:
      --i0=<VALUE>           Forward scattering intensity
      --rg=<VALUE>           Experimental Radius of Gyration
      --rho=<VALUE>          Average protein density (default: 1.37 g/cm^-3)
  -a, --offset=<VALUE>       Offset coefficient (default: look up)
  -b, --scaling=<VALUE>      Scaling coefficient (default: look up)
  -h, --help                 Print usage information and exit
  -v, --version              Print version information and exit

Mandatory arguments to long options are mandatory for short options too.


$ datmow --version
datmow, ATSAS 3.0.1 (r12314)
Copyright (c) ATSAS Team, EMBL, Hamburg Outstation 2014-2020

$ datmow --help
Usage: datmow [OPTIONS] [FILE(S)]

Molecular weight from Apparent Volume (Fischer et al., 2010).
Output: smax (A^-1), Q', V' (apparent volume), V (Volume, A^3), MW (Da), file name

Known Arguments:
  FILE                       Data file

Known Options:
      --i0=<VALUE>           Forward scattering intensity
      --rg=<VALUE>           Experimental Radius of Gyration
      --first=<N>            index of the first point to be used (default: 1)
  -h, --help                 Print usage information and exit
  -v, --version              Print version information and exit

Mandatory arguments to long options are mandatory for short options too.


"""

def atsas_dat_tools(fn_out, path=""):
    # datporod: the used Rg, I0, the computed volume estimate and the input file name
    #
    # datvc: the first three numbers are the integrated intensities up to 0.2, 0.25 and 0.3, respectively. 
    #        the second three numbers the corresponding MW estimates
    #
    # datmow: Output: Q', V' (apparent Volume), V (Volume, A^3), MW (Da), file name
    ret = run(["datporod", fn_out], path).split('\n')
    try:
        Vv = extract_vals(ret[0])[-1]
        r_porod = {"vol": Vv}
    except:
        r_porod = {"vol": np.nan}
        print("Unable to get output from datporod ...")
    
    #ret = run(f"datvc {fn_out}").split('\n')
    #try:
    #    ii1,ii2,ii3,mw1,mw2,mw3 = extract_vals(ret[0])
    #    r_vc = {"MW": [mw1, mw2, mw3]}
    #except:
    #    print("Unable to get output from datvc ...")
    
    ret = run(["datmow", fn_out], path).split('\n')
    try:
        Qp,Vp,Vv,mw = extract_vals(ret[0])[-4:]
        r_mow = {"Q": Qp, "app vol": Vp, "vol": Vv, "MW": mw}
    except:
        r_mow = {"Q": np.nan, "app vol": np.nan, "vol": np.nan, "MW": np.nan}
        print("Unable to get output from datmow ...")

    return {"datporod": r_porod, 
            #"datvc": r_vc,  # this won't work if q_max is below 0.3 
            "datmow": r_mow}


def gen_atsas_report(d1s, ax=None, fig=None, sn=None, skip=0, q_cutoff=0.6, 
                     plot_full_q_range=False, print_results=True, path=""):
    if not os.path.isdir("processed"):
        os.mkdir("processed")
    
    if ax is None:
        ax = []
        if fig is None:
            fig = plt.figure(figsize=(9,3))
        # rect = l, b, w, h
        ax.append(fig.add_axes([0.09, 0.25, 0.25, 0.6])) 
        ax.append(fig.add_axes([0.41, 0.25, 0.25, 0.6])) 
        ax.append(fig.add_axes([0.73, 0.25, 0.25, 0.6])) 
        ax.append(ax[0].twiny())
    else:
        #ax[0].figure.cla()
        for a in ax:
            a.clear()
    
    if sn is None:
        tfn = "processed/t.dat"
        tfn_out = "processed/t.out"
    else:
        tfn = "processed/t.dat"
        tfn_out = f"processed/{sn}.out"
    
    sk0 = skip
    qc0 = q_cutoff 
    if skip<0:
        sk0 = 0
    if q_cutoff<0:
        qc0 = 0.3
    atsas_create_temp_file(tfn, d1s, skip=sk0, q_cutoff=qc0)

    re_autorg = atsas_autorg(tfn, path=path)
    if re_autorg["Rg"]==0: # autorg not successful,py4xs.slnxs might work
        re_autorg["I0"],re_autorg["Rg"],re_autorg["fit range"] = d1s.plot_Guinier(no_plot=True)
        
    if skip<0:
        sk0 = re_autorg["fit range"][0]
    if q_cutoff<0 and re_autorg["Rg"]>0:
        qc0 = 15./re_autorg["Rg"]

    re_gnom = atsas_datgnom(tfn, re_autorg["Rg"], first=sk0+1,
                            last=len(d1s.qgrid[d1s.qgrid<=qc0]), fn_out=tfn_out, path=path)
    try:
        hdr,dq,di,dr,dpr,dpre = read_gnom_out_file(tfn_out)
    except: # this would happen if gnom fails to run
        hdr,dq,di,dr,dpr,dpre = 0,0,0,0,0,0
        
    if plot_full_q_range:
        idx = (dq>=d1s.qgrid[0])
        ax[0].loglog(dq[idx], di[idx], zorder=2)
        ax[0].errorbar(d1s.qgrid, d1s.data, d1s.err, fmt=".", alpha=0.3, zorder=1)
    else:
        idx = (d1s.qgrid<qc0)
        ax[0].semilogy(dq, di, zorder=2)
        ax[0].errorbar(d1s.qgrid[idx], d1s.data[idx], d1s.err[idx], fmt=".", alpha=0.3, zorder=1)
    #ax[0].yaxis.set_major_formatter(plt.NullFormatter())
    #ax[0].set_title("intensity")
    ax[0].set_xlabel(r"$q$")
    if re_autorg["Rg"]>0 and di is not None:
        Rg = re_autorg["Rg"] 
        I0 = re_autorg["I0"]
        n1,n2 = re_autorg["fit range"]
        qf = d1s.qgrid[n1:n2]
        axx = ax[3]
        if plot_full_q_range and Rg<50:
            gf = I0*np.exp(-(qf*Rg)**2/3)/10
            idx = (d1s.qgrid<1.5/re_autorg["Rg"])
            axx.plot(qf*qf, gf, "k", zorder=2)
            #axx.errorbar(d1s.qgrid[n1:n2]**2, d1s.data[n1:n2]/10, d1s.err[n1:n2], fmt="b.", alpha=0.6, zorder=1)
            axx.errorbar(d1s.qgrid[idx]**2, d1s.data[idx]/10, d1s.err[idx]/10, fmt="b.", alpha=0.6, zorder=1)
            qm2 = (1.5/re_autorg["Rg"])**2
            axx.set_xlim(0, 4*qm2)
            locs = axx.get_xticks()
            locs = locs[locs<=2*qm2]
            if len(locs>6):
                locs = locs[::2]
            labels = [str(t) for t in locs]
            axx.set_xticks(locs) #[:-2])
            axx.set_xticklabels(labels) #[:-2])
            axx.set_xlabel(r"$q^2$", loc='right')
        else:
            qf = d1s.qgrid[n1:n2]
            gf = I0*np.exp(-(qf*Rg)**2/3)
            idx = (d1s.qgrid<1.5/re_autorg["Rg"])
            axx.plot(qf*qf, gf, "k", zorder=2)
            #axx.errorbar(d1s.qgrid[n1:n2]**2, d1s.data[n1:n2], d1s.err[n1:n2], fmt="b.", alpha=0.6, zorder=1)
            axx.errorbar(d1s.qgrid[idx]**2, d1s.data[idx], d1s.err[idx], fmt="b.", alpha=0.6, zorder=1)
            qm2 = (1.5/re_autorg["Rg"])**2
            axx.set_xlim(-qm2, 1.2*qm2)
            locs = axx.get_xticks()
            if len(locs>6):
                locs = locs[::2]
            labels = [str(t) for t in locs if t>=0]
            axx.set_xticks(locs[locs>=0])
            axx.set_xticklabels(labels)
            axx.set_xlabel(r"$q^2$", loc='left')
    
    if re_autorg["Rg"]==0:
        kratky_qm=0.3
        idx = (d1s.qgrid<kratky_qm)
        ax[1].plot(d1s.qgrid[idx], d1s.data[idx]*np.power(d1s.qgrid[idx], 2))
        ax[1].set_xlabel(r"$q$")
    else:
        kratky_qm=10./re_autorg["Rg"]
        idx = (d1s.qgrid<kratky_qm)
        ax[1].plot(d1s.qgrid[idx]*re_autorg["Rg"], d1s.data[idx]*np.power(d1s.qgrid[idx], 2))
        ax[1].set_xlabel(r"$q \times R_g$")    
    ax[1].yaxis.set_major_formatter(plt.NullFormatter())
    ax[1].set_title("kratky plot")

    ax[2].errorbar(dr, dpr, dpre)
    ax[2].yaxis.set_major_formatter(plt.NullFormatter())
    ax[2].set_title(r"$P(r)$")
    ax[2].set_xlabel(r"$r$")

    ret = atsas_dat_tools(tfn_out, path=path)
    if print_results:
        print(f"Gunier fit: quality = {re_autorg['quality']} %,", end=" ")
        print(f"I0 = {re_autorg['I0']:.2f} +/- {re_autorg['I0 err']:.2f} , ", end="")
        print(f"Rg = {re_autorg['Rg']:.2f} +/- {re_autorg['Rg err']:.2f}")
        print(f"GNOM fit: quality = {re_gnom['quality']:.2f}, Dmax = {re_gnom['Dmax']:.2f}, Rg = {re_gnom['Rg (r)']:.2f}")
        print(f"Volume estimate: {ret['datporod']['vol']:.1f} (datporod), {ret['datmow']['vol']:.1f} (MoW)")
        print(f"MW estimate: {ret['datmow']['MW']/1000:.1f} kDa (MoW)")          
    else:
        txt = f"Gunier fit: quality = {re_autorg['quality']} %, "
        txt += f"I0 = {re_autorg['I0']:.2f} +/- {re_autorg['I0 err']:.2f} , "
        txt += f"Rg = {re_autorg['Rg']:.2f} +/- {re_autorg['Rg err']:.2f}\n"
        txt += f"GNOM fit: quality = {re_gnom['quality']:.2f}, Dmax = {re_gnom['Dmax']:.2f}, Rg = {re_gnom['Rg (r)']:.2f}\n"
        txt += f"Volume estimate: {ret['datporod']['vol']:.1f} (datporod), {ret['datmow']['vol']:.1f} (MoW)\n"
        txt += f"MW estimate: {ret['datmow']['MW']/1000:.1f} kDa (MoW)"          
        return txt
    
def gen_pdf_report(fn):
    """ create a pdf file to summarize the static solution scattering data in the specified h5 file
    """
    dn = os.path.dirname(fn)
    if dn == "":
        dn = "."

    bn = os.path.basename(fn).split('.')
    if bn[-1]!='h5':
        raise Exception(f"{bn} does not appear to be a h5 file.")
    bn = bn[0]
    
    pn = lixtools.__path__
    if isinstance(pn, list):
        pn = pn[0]

    tmp_dir = tempfile.gettempdir()
    fn0 = os.path.join(pn, f"template_report.ipynb")
    fn1 = os.path.join(dn, f"{bn}_report.ipynb")
        
    print("preparing the notebook ...")
    ret = run(["cp", fn0, fn1])
    # sed only works with unix
    #ret = run(["sed", "-i", f"s/00template00/{bn}/g", fn1])
    with open(fn1, 'r+') as fh:
        txt = fh.read()
        txt = re.sub('00template00.h5', fn, txt)
        fh.seek(0)
        fh.write(txt)
        fh.truncate()
    
    fn2 = os.path.join(tmp_dir, f"{bn}_report.pdf")
    print("executing ...")
    ret = run(["jupyter", "nbconvert", 
               fn1, f"--output={fn2}", 
               "--ExecutePreprocessor.enabled=True", 
               "--TemplateExporter.exclude_input=True", "--to", "pdf"],
              debug=True)    
    print("cleaning up ...")
    ret = run(["mv", fn2, dn])
    ret = run(["rm", fn1])

