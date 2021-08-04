#!/usr/bin/python3

import cv2,os,subprocess
import socket,time,json
from pyzbar.pyzbar import decode,ZBarSymbol
import numpy as np
from itertools import product

def look_for_device(cam_name):
    ret = subprocess.run(["v4l2-ctl", "--list-devices"], stdout=subprocess.PIPE)
    for _ in ret.stdout.decode().split('\n\n'):
        if cam_name in _:
            for __ in _.split("\t"):
                if "/dev" in __:
                    return __.strip("\n")
    
    print("could not find", cam_name)
    return None

class webcam:
    # adapted from pypi/camset
    def __init__(self, device=None, cam_name=None):
        self.cap = None
        if cam_name:
            device = look_for_device(cam_name)
        self.device = device
        self.res_list = self.read_resolution_capabilites()
        
    def read_resolution_capabilites(self):
        if not self.device:
            print("the camera is not connected.")
            return None
        
        outputread = subprocess.run(['v4l2-ctl', '-d', self.device, '--list-formats-ext'], 
                                    check=True, universal_newlines=True, stdout=subprocess.PIPE)
        outputs = outputread.stdout.split('\n')
        res_list = []
        pre = ''
        post = ''
        for line in outputs:
            if ":" in line:
                line = line.strip()
                if "'" in line:
                    pre = line.split("'", 1)[1]
                    pre = pre.split("'", 1)[0]
                else:
                    if "Size:" in line:
                        post = line.split("Size: ", 1)[1]
                        post = post.split(" ")[-1]
                        output = " - ".join((pre, post))
                        res_list.append(output)
        return res_list

    def start_camera_feed(self, pixelformat, vfeedwidth, vfeedheight):
        if not self.device:
            print("the camera is not connected.")
            return 
        
        cmd = ['v4l2-ctl', '-d', self.device, 
               '-v', 'height={0},width={1},pixelformat={2}'.format(vfeedheight, vfeedwidth, pixelformat)]
        subprocess.run(cmd, check=True, universal_newlines=True)
        time.sleep(1)

        cmd = ['v4l2-ctl', '-d', self.device,
               '-c', 'focus_auto=0,exposure_auto=1']
        subprocess.run(cmd, check=True, universal_newlines=True)

        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap:
            raise Exception(f"unable to capture device {self.device}")
        
        # also set resolution to cap, otherwise cv2 will use default and not the res set by v4l2
        self.cap.set(3,int(vfeedwidth))
        self.cap.set(4,int(vfeedheight))
        fourcode = cv2.VideoWriter_fourcc(*f'{pixelformat}')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcode)

    def set_focus(self, focus=50):
        if not self.device:
            print("the camera is not connected.")
            return 
        
        subprocess.run(['v4l2-ctl', '-d', self.device,
                        '-c', ("focus_absolute=%d" % focus)],
                       check=True, universal_newlines=True)

    def set_zoom(self, zoom=100):
        if not self.device:
            print("the camera is not connected.")
            return 
        
        subprocess.run(['v4l2-ctl', '-d', self.device,
                        '-c', ("zoom_absolute=%d" % zoom)],
                       check=True, universal_newlines=True)

    def set_exposure(self, exp=250):
        if not self.device:
            print("the camera is not connected.")
            return

        subprocess.run(['v4l2-ctl', '-d', self.device,
                        '-c', ("exposure_absolute=%d" % exp)],
                       check=True, universal_newlines=True)

    def stop_camera_feed(self):
        if not self.device:
            print("the camera is not connected.")
            return
        
        self.cap.release()
        
    def snapshot(self, rep=5):
        for _ in range(rep):  # make sure that the image is updated
            ret,img = self.cap.read()
        
        return img

    def snapshot_w_avg(self, n=10, skip=2):
        imgs = [self.snapshot(rep=2) for _ in range(n)][skip:]
        return np.asarray(np.average(imgs, axis=0), dtype=np.uint8)
    
    
def read_code(img, target="3QR", crop_x=1, crop_y=1, debug=True, fn="cur.png"):
        h,w = img.shape[:2]
        x1 = int(w*(1-crop_x)/2)
        x2 = int(w*(1+crop_x)/2)
        y1 = int(h*(1-crop_y)/2)
        y2 = int(h*(1+crop_y)/2)

        f1 = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        ret = {}
        for th,bkg in product(range(21,35,4), range(1,9,2)):
            f4 = cv2.adaptiveThreshold(f1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, th, bkg)
            d = decode(f4, symbols=[ZBarSymbol.QRCODE])
            for i in range(len(d)):
                k = d[i].data.decode()
                if not k in ret.keys():
                    ret[k] = d[i]
        cs = list(ret.values())
        
        cv2.imwrite(f"img/raw-{fn}", f1)

        if target=="3QR": # expecting 3x QR codes 
            cs = decode(f4, symbols=[ZBarSymbol.QRCODE]) 
            code = {}
            for cc in cs:
                xi = 2-int(cc.rect.left/400) # has to so with the orientation of the camera
                code[chr(xi+ord('a'))] = cc.data.decode("utf-8") 
        elif target=="1QR": # 1x QR code
            cs = decode(f4, symbols=[ZBarSymbol.QRCODE]) 
            if len(cs)!=1:
                print(f"ERR: {len(cs)} code(s) were read, expecting 1")
                code = [] 
            else:
                code = [cs[0].data.decode("utf-8")]
        elif target=="1bar": # 1x code128 barcode
            cs = decode(f4, symbols=[ZBarSymbol.CODE128]) 
            if len(cs)!=1:
                print(f"ERR: {len(cs)} code(s) were read, expecting 1")
                code = []
            else:
                code = [cs[0].data.decode("utf-8")]

        if debug:
            print(cs)
        return json.dumps(code)
    

def run():
    
    opentron_sock_port = 9999

    cam = webcam(cam_name="Logitech Webcam C930e")
    cam.start_camera_feed("YUYV", 1280, 720)
    cam.set_zoom(350)
    cam.set_focus(50)
    
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('10.16.0.10', opentron_sock_port))
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.listen(5)
    print('listening ...')

    while True:
        clientsocket,addr = serversocket.accept()
        print(f"{time.asctime()}: got a connection from {addr} ...")
        msg = clientsocket.recv(128).decode()
       
        print(msg)
        # this should be a json encoded dictionary
        # required keys: target
        # optional keys: focus, zoom, crop_x, crop_y
        cmd = json.loads(msg)
        target = cmd['target']
        if target in ["3QR", "1QR", "1bar"]:
            if "zoom" in cmd.keys():
                cam.set_zoom(cmd['zoom'])
            if "focus" in cmd.keys():
                cam.set_focus(cmd['focus'])
            if "exposure" in cmd.keys():
                cam.set_exposure(cmd['exposure'])
            crop_x = (cmd["crop_x"] if "crop_x" in cmd.keys() else 1)
            crop_y = (cmd["crop_y"] if "crop_y" in cmd.keys() else 1)
            pstr = ("%02d-"%cmd["slot"] if "slot" in cmd.keys() else "")
            img = cam.snapshot()  # read some images to make sure new camera settings take effect
            img = cam.snapshot()
            code = read_code(img, target=target, 
                             crop_x=crop_x, crop_y=crop_y, 
                             debug=True, fn=f"{pstr}{target}.png")
        else:
            print("invalid request.")
            
        clientsocket.send(code.encode("ascii"))
        time.sleep(0.1)
        clientsocket.close()

    
    
if __name__ == "__main__":
    run()
