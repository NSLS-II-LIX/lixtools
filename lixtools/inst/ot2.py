#!/usr/bin/python3.7
#
# this is the script that runs on the OT2 RasPi to read plate/holder QR codes
# it talks to the webcam "server" running on the BeagleBone 
# 

import sys, getopt
from opentrons import robot, labware, instruments
from opentrons.util import environment
import opentrons,json
from opentrons.protocol_api import labware as lbw

from threading import Thread
from collections import Counter,OrderedDict
import socket,time
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


lix_holder_offset = (70, 52, 170)
flat96_offset = (130, 52, 170)


def read_code(sock, target, slot):
    if target=="1bar":
        cmd = {"target": "1bar", "zoom": 200, "focus": 45, "slot": slot}
    elif target=="1QR":
        cmd = {"target": "1QR", "zoom":200, "focus": 45, 
               "crop_x": 0.3, "crop_y": 0.4, "slot": slot, "exposure": 40} 
    elif target=="3QR":
        cmd = {"target": "3QR", "zoom": 350, "focus": 50, 
               "crop_y": 0.4, "exposure": 35, "slot": slot}
    else:
        return []
    
    ret = sock.send_msg(json.dumps(cmd))
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
               "plate_type" : '96-flat',
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
    
    pdict = OrderedDict()
    hdict = OrderedDict()
    ldict = {"plates": pdict, "holders": hdict}
    tips = []

    sc = sock_client(('192.168.7.2', code_reader_sock_port))   # BeagleBone
    
    robot.connect()
    robot.turn_on_rail_lights()
    robot.home()
    p300 = instruments.P300_Single(mount='left')
    
    retry = 3
    for pos,lbw in config.items():
        lbware = labware.load("96-flat", pos) #, share=True) # had trouble with slot2 without sharing
        if lbw=="96-flat":
            robot.move_to((lbware, flat96_offset), p300)
            ret = read_code(sc, "1QR", pos)
            for _ in range(retry): 
                if len(ret)==1:
                    pdict[ret[0]] = {"slot": pos}
                    print(pos, ret[0])
                    break
        elif lbw in ['lix_3x_holder', 'lix_3x_holder_c']:
            robot.move_to((lbware, lix_holder_offset), p300)
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

    robot.turn_off_rail_lights()
    robot.home()
    robot.disconnect()
    
    print("*****", json.dumps(ldict))

    
if __name__ == "__main__":
   main(sys.argv[1:])