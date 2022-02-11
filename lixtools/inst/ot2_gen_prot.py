#!/nsls2/xf16id1/sw/conda_envs/analysis/bin/python
from lixtools.mailin import generate_docs
import sys,os,getopt
from lixtools.mailin import generate_docs

def print_help():
    print("ot2_gp <options> spread_sheet(s)")
    print("Specify the location of tips (-t, --tips=), holders (-h, --holders=), and plates (-p, --plates=).")
    print("Oprionally specify run_name (-n, --run_name=, default is 'test')")
    print("                   max # of samples between buffers (-l, --buf_limit=, default is 4)")
    print("                   aspirate flow rate (-r, --aspirate_fr=, default is 20)")
    print("                   bottom clearance on aspirate (-c, --bottom_clearance=, default is 0.5)")
    print("                   pause between mxing and transfer (-z, --pause_after_mixing=, default is Y)")        

def main(argv):
    try:
        optlist, args = getopt.getopt(argv, "p:t:h:n:l:r:c:z", 
                                      ["tips=", "plates=", "holders=", "run_name=", 
                                       "buf_limit=", "aspirate_fr=", "bottom_clearance=", "pause_after_mixing="])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    
    if len(args)==0 or len(optlist)==0:
        print_help()
        exit(0)
    
    xls_fns = args
    for fn in xls_fns:
        if not os.path.isfile(fn):
            raise Exception(f"{fn} does not exist.")
    
    ot2_layout = {}
    parms = dict(run_name="test", b_lim=4, flow_rate_aspirate = 20,
                 bottom_clearance=0.5, pause_after_mixing=True)
    for opt,v in optlist:
        if opt in ["-t", "--tips"]:
            ot2_layout["tips"] = v
        elif opt in ["-p", "--plates"]:
            ot2_layout["plates"] = v
        elif opt in ["-h", "--holders"]:
            ot2_layout["holders"] = v
        elif opt in ["-n", "--run_name"]:
            parms["run_name"] = v
        elif opt in ["-l", "--buf_limit"]:
            parms['b_lim'] = v
        elif opt in ["-r", "--aspirate_fr"]:
            parms['flow_rate_aspirate'] = v
        elif opt in ["-c", "--bottom_clearance"]:
            parms['bottom_clearance'] = v
        elif opt in ["-z", "--pause_after_mixing"]:
            parms['pause_after_mixing'] = (v=='Y')
            
    if len(ot2_layout.keys())<3:
        raise Exception(f"Incomplete specification of OT2 layout: {ot2_layout}")
    
    print(ot2_layout, parms)        
    generate_docs(ot2_layout, xls_fns, **parms)
        
if __name__ == "__main__":
   main(sys.argv[1:])

