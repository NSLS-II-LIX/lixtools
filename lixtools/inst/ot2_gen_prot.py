import sys,os,getopt,socket,time
import pandas as pd
import numpy as np
from itertools import groupby,chain

class sock_client:
    def __init__(self, sock_addr, persistent=True):
        self.sock_addr = sock_addr
        self.delay = 0.05
        self.sock = None
        self.persistent = persistent

        if persistent:
            self.connect()
    
    def clean_up(self):
        if self.sock is not None:
            self.sock.close()

    def connect(self, timeout=-1):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        ts = time.time()
        while True:
            try:
                self.sock.connect(self.sock_addr)   
                print("connected.")
                break
            except ConnectionRefusedError:
                if timeout>0 and time.time()-ts>timeout:
                    raise f"unable to connect after {timeout} sec." 
                time.sleep(1)
                print(f"{time.asctime()}: waiting for a connection ...   \r", end="")
        
    def send_msg(self, msg):
        if not self.persistent:
            self.sock.connect(self.sock_addr)
        self.sock.send(msg.encode())
        time.sleep(self.delay)
        ret = self.sock.recv(8192).decode('ascii')
        if not self.persistent:
            self.sock.close()
        return ret


def read_code(cam, target, slot):
    if target=="1bar":
        cmd = {"target": "1bar", "zoom": 200, "focus": 45, "slot": slot}
    elif target=="1QR":
        cmd = {"target": "1QR", "zoom": 200, "focus": 45, 
               "crop_x": 0.3, "crop_y": 0.6, "slot": slot, "exposure": 30} 
    elif target=="3QR":
        cmd = {"target": "3QR", "zoom": 300, "focus": 55, 
               "crop_y": 0.6, "exposure": 25, "slot": slot}
    else:
        return []
    
    ret = cam.process_cmd(cmd)
    print(ret)
    return json.loads(ret)


def read_OT2_layout2(plate_slots, holder_slots, retry=3,
                     camserv_ip="169.254.246.60", camserv_port=9999,
                     ot2_ssh_ip = "169.254.246.65", ot2_ssh_port=22,
                     ot2_cmd_ip="169.254.246.65", ot2_cmd_port=9999, timeout=60):
    """ this runs check_deck_config2.py to move the gantry
        but needs to communicate to the camserv separately to read the code
    """
    ssh_key = str(pathlib.Path.home())+"/.ssh/ot2_ssh_key"
    if not os.path.isfile(ssh_key):
        raise Exception(f"{ssh_key} does not exist!")

    print("starting script on OT2 ...")
    cmd = ["ssh", "-i", ssh_key, "-o", f"port={ot2_ssh_port}",
           f"root@{ot2_ssh_ip}", "/var/lib/jupyter/notebooks/check_deck_config2.py"]
    #proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    th = threading.Thread(target=subprocess.run, args=(cmd,))
    th.start()

    print("connect to webcam ...")
    # establish connection to the camserv
    cam = webcam(cam_name="Logitech Webcam C930e")
    cam.start_camera_feed("YUYV", 1280, 720)
    cam.set_zoom(350)
    cam.set_focus(50)
    if not os.path.exists('img'):
        os.makedirs('img')

    print("connecting to OT2 ...")
    # OT2 socket, won't be ready until the gantry is homed
    ot2_sock = sock_client((ot2_cmd_ip, ot2_cmd_port))

    pdict = OrderedDict()
    hdict = OrderedDict()
    ldict = {"plates": pdict, "holders": hdict}

    for ps in plate_slots.split(','):
        ot2_sock.send_msg(f"{ps},plate")
        ps = int(ps)
        for _ in range(retry): 
            ret = read_code(cam, "1QR", ps)
            if len(ret)==1:
                pdict[ret[0]] = {"slot": ps}
                print(ps, ret[0])
                break

    for ps in holder_slots.split(','):
        ot2_sock.send_msg(f"{ps},holder")
        ps = int(ps)
        for _ in range(retry): 
            ret = read_code(cam, "3QR", ps)
            if len(ret.keys())==3:
                break
        for hidx in ['a', 'b', 'c']:
            if hidx in ret.keys():
                hdict[ret[hidx]] = {"slot": ps, "holder": hidx}
        print(ps, ret)        
    
    ot2_sock.send_msg("done")
    ot2_sock.clean_up()
    th.join()

    return ldict

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
        
    # create tranfer list
    mixing_list = []
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
            blow_out = False
            if wn in mdict.keys():
                # mix first
                for ws,v in mdict[wn].items():
                    mixing_list.append([pname, ws, pname, wn, [v, blow_out]])
                    if not blow_out:
                        blow_out = True
                    vt += v
                mixing_list[-1][-1][-1] = vt
                    
            # now transfer
            transfer_list.append([pname, wn, hname, hpos, vt])
            
            # add entry into the sample dictionary
            slist.append([hname, hpos, wdict[wn]["Sample"], wdict[wn]["Buffer"], vt])

    return slist,transfer_list,bdict,mixing_list
    
def read_OT2_layout(plate_slots, holder_slots, msg=None, ot2_ip="169.254.246.65", ot2_port=22):
    """ 
    the arguments should be a comma-separated list of slot positions on the Opentron deck
    e.g. "1,2"  
    revise IP and port when using ssh tunnel to connect to OT2
    """
    if msg is None:
        ssh_key = str(pathlib.Path.home())+"/.ssh/ot2_ssh_key"
        if not os.path.isfile(ssh_key):
            raise Exception(f"{ssh_key} does not exist!")

        cmd = ["ssh", "-i", ssh_key, "-o", f"port={ot2_port}",
               f"root@{ot2_ip}", "/var/lib/jupyter/notebooks/check_deck_config.py", 
               "-h", holder_slots, "-p", plate_slots]

        ret = subprocess.run(cmd, capture_output=True)
        msg = ret.stdout.decode()

        if ret.returncode:
            print(msg)
            print(ret.stderr.decode())
            raise Exception("error executing check_deck_config.py")
            
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
                  plate_type = "biorad_96_wellplate_200ul_pcr",
                  holder_type = "lix_3x_holder_c",
                  tip_type = "opentrons_96_tiprack_300ul",
                  flow_rate_aspirate = 20, flow_rate_dispense = 50, 
                  n_mix=3, pause_after_mixing=True,
                  bottom_clearance = 0.5
                 ):
    """ ot2_layout should be a dictionary:
            {"plates" : "1,2",
             "holders" : "7,8",
             "tips" : "9,10"}
    """
    print("Processing sample list(s) ...")
    slist,transfer_list,bdict,mixing_list = process_sample_lists(xls_fns, b_lim=b_lim)
    
    if ldict is None:
        print("Reading bar/QR codes, this might take a while ...")
        ldict = read_OT2_layout2(ot2_layout["plates"], ot2_layout["holders"])
    
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
                "            'apiLevel': '2.9'\n",
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

    if pause_after_mixing:
        ops_list = mixing_list+["pause"]+transfer_list
    else:
        ops_list = mixing_list+transfer_list
    
    bl_str = "blow_out=True, blowout_location='destination well'"
    for st in ops_list:
        if st=="pause":
            protocol.append(f"    ctx.pause('Mixing completed. Resume to start transfer.')\n")
            continue
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

        if isinstance(vol, list):
            # part of mixing
            vol,action = vol  
            #protocol.append(f"    pipet.transfer({vol}, {sname}, {dname})\n")
            #if action==True:
            #    protocol.append(f"    pipet.blow_out()\n")
            if action>1: # mixing 
                protocol.append(f"    pipet.transfer({vol}, {sname}, {dname}, mix_after=({n_mix}, {action}), {bl_str})\n")
            else:
                protocol.append(f"    pipet.transfer({vol}, {sname}, {dname}, {bl_str})\n")
        else:
            protocol.append(f"    pipet.transfer({vol}, {sname}, {dname}, {bl_str})\n")

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
                "            'apiLevel': '2.9'\n",
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

