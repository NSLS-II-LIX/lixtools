#!/usr/bin/python3.7
## as found on OT2 2023/01/20

import os,sys,getopt,json

from threading import Thread
from collections import Counter,OrderedDict
import socket,time

import opentrons.execute
from opentrons import types

cam_server = '169.254.246.60'
#cam_server = '192.168.7.2'  # BB
code_reader_sock_port = 9999

class sock_client:
    def __init__(self, sock_addr):
        self.sock_addr = sock_addr
        self.delay = 0.05
    
    def clean_up(self):
        if self.sock is not None:
            self.sock.close()

    def send_msg(self, msg):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(self.sock_addr)   
        sock.send(msg.encode('ascii'))
        time.sleep(self.delay)
        ret = sock.recv(8192).decode('ascii')
        sock.close()
        return ret


def read_code(sock, target, slot):
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
    
    ret = sock.send_msg(json.dumps(cmd))
    print(json.dumps(cmd))
    print(ret)
    return json.loads(ret)
    


def main(argv):
    try:
        optlist, args = getopt.getopt(argv, "h:p:t:", ["tips=", "plates=", "holders=", "help"])
    except getopt.GetoptError:
        print("Specify the location of tips (-t, --tips), holders (-h, --plates), and plates (-p, --plates).")
        sys.exit(2)
        
    config = {}
    lw_type = {"holder_type": 'lix_3x_holder_c',
               "plate_type" : 'corning_96_wellplate_360ul_flat',
               "tip_rack": 'opentrons-tiprack-300ul'}
    
    for opt,arg in optlist:
        if opt in ["--tip_rack", "--plate_type", "--holder_type"]:
            lw_type[opt[2:]] = arg

    for opt,arg in optlist:
        if opt=="--help":
            print("Specify the location of tips (-t, --tips), holders (-h, --plates), and plates (-p, --plates).")
            print("Can also specify the type of tiprack (--tip_rack) and plate (--plate_type). Must be valid names.")
            print("Default values are opentrons-tiprack-300ul and 96-flat.")
        elif opt=="-t" or opt=="--tips":
            for pos in arg.split(","):
                config[pos] = lw_type["tip_rack"]
        elif opt=="-h" or opt=="--holders":
            for pos in arg.split(","):
                config[pos] = lw_type["holder_type"]
        elif opt=="-p" or opt=="--plates":
            for pos in arg.split(","):
                config[pos] = lw_type["plate_type"]
    
    #ldict = {"plates": {}, "holders": {}}
    pdict = OrderedDict()
    hdict = OrderedDict()
    ldict = {"plates": pdict, "holders": hdict}
    tips = []

    sc = sock_client((cam_server, code_reader_sock_port))   # BeagleBone
    
    #os.system("systemctl stop opentrons-robot-server") # otherwise the lights won't turn on
    protocol = opentrons.execute.get_protocol_api('2.9')
    protocol.home()
    protocol.set_rail_lights(True)
    pipet = protocol.load_instrument('p300_single', 'left')    

    retry = 3
    for pos,lbw in config.items():
        lbware = protocol.load_labware("corning_96_wellplate_360ul_flat", pos)
        
        if lbw=="corning_96_wellplate_360ul_flat":
            pipet.move_to(lbware["D12"].center().move(types.Point(x=10, y=0, z=230)))
            ret = read_code(sc, "1QR", pos)
            for _ in range(retry): 
                if len(ret)==1:
                    pdict[ret[0]] = {"slot": pos}
                    print(pos, ret[0])
                    break
        elif lbw in ['lix_3x_holder', 'lix_3x_holder_c']:
            pipet.move_to(lbware["D7"].center().move(types.Point(x=0, y=0, z=180)))
            for _ in range(retry): 
                ret = read_code(sc, "3QR", pos)
                if len(ret.keys())==3:
                    break
            for hidx in ['A', 'B', 'C']:
                if hidx in ret.keys():
                    hdict[ret[hidx]] = {"slot": pos, "holder": hidx}
            print(pos, ret)
        elif "tiprack" in lbw:
            tips.append(lbware)

    protocol.set_rail_lights(False)
    protocol.home()
    os.system("systemctl start opentrons-robot-server") # otherwise the lights won't turn on
    
    print("*****", json.dumps(ldict))
    
    #return plates,holders,tips                 
                
if __name__ == "__main__":
   main(sys.argv[1:])
