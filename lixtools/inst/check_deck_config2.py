#!/usr/bin/python3.7
import os,sys,getopt,json

from threading import Thread
from collections import Counter,OrderedDict
import socket,time

import opentrons.execute
from opentrons import types

OT2_IP = '169.254.246.65'
OT2_sock_port = 9999

def main(argv):
    print("Waking up, this might take some time ...")
    os.system("systemctl stop opentrons-robot-server") # otherwise the lights won't turn on
    protocol = opentrons.execute.get_protocol_api('2.9')
    protocol.home()
    protocol.set_rail_lights(True)
    pipet = protocol.load_instrument('p300_single', 'left')    
    print("Ready to move ...")

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind((OT2_IP, OT2_sock_port))
    serversocket.listen(5)
    print('listening ...')

    clientsocket,addr = serversocket.accept()
    print(f"{time.asctime()}: got a connection from {addr} ...")
    while True:
        msg = clientsocket.recv(128).decode().strip('\n\r')
        if not msg:
            continue
        elif msg=="done":
            break    

        pos,lbw = msg.split(',')
        lbware = protocol.load_labware("corning_96_wellplate_360ul_flat", pos)
        print(f"moving to pos {pos}")        

        if lbw=="plate":
            pipet.move_to(lbware["D12"].center().move(types.Point(x=10, y=0, z=230)))
        elif lbw=="holder":
            pipet.move_to(lbware["D7"].center().move(types.Point(x=0, y=0, z=180)))

        clientsocket.sendall("ready".encode())

    protocol.set_rail_lights(False)
    #protocol.home()
    #os.system("systemctl start opentrons-robot-server") # otherwise the lights won't turn on
    
    print("*****")
                
if __name__ == "__main__":
   main(sys.argv[1:])
