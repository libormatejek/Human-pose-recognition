import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import sys
import json
import math

def cmp(a,b,cmin,cmax):	
    ma = (a[:,:,0] >= cmin[0]) & (a[:,:,0] <= cmax[0]) & (a[:,:,1] >= cmin[1]) & (a[:,:,1] <= cmax[1]) & (a[:,:,2] >= cmin[2]) & (a[:,:,2] <= cmax[2])
    mb = (b[:,:,0] >= cmin[0]) & (b[:,:,0] <= cmax[0]) & (b[:,:,1] >= cmin[1]) & (b[:,:,1] <= cmax[1]) & (b[:,:,2] >= cmin[2]) & (b[:,:,2] <= cmax[2])

    #Image.fromarray(b, 'RGB').show()
    #cv2.waitKey()

    sum0 = np.count_nonzero(ma);
    sum1 = np.count_nonzero(mb);
    sum2 = np.count_nonzero(np.logical_and(ma,mb));
    sum3 = np.count_nonzero(np.logical_and(np.logical_not(ma),mb))             #np.count_nonzero(np.logical_xor(ma,mb));

    if sum0==0:
        res = None
    else:
        p1 = sum2/sum0;
        p2 = sum3/sum0;
        res=p1
    return (sum0,sum1,sum2,sum3,res)
def readFile(name):	
    im = cv2.imread(name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return np.asarray(im)
def getLength(p1,p2):
    return pow(pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2),0.5)
def getCenter(p1,p2):
    return ((p1+p2)/2).astype("i")
def drawLine(p1,p2,h,image,rgb):
    cv2.line(image, tuple(p1), tuple(p2), tuple(rgb) , max(int(h),1) , lineType=cv2.LINE_AA)
def deflate(p1,p2,k):
    dv = (p2 - p1)*k*0.5
    dv = dv.astype("i")
    return (p1+dv,p2-dv)
def rotate(p1,p2,fi):
    v = p2-p1
    v = np.matmul(np.array([[math.cos(fi),-math.sin(fi)],[math.sin(fi),math.cos(fi)]]),v)
    return (p1+v).astype("i")
def drawBodyParts(name,kp,image0,k):
    #image = np.ones(image0.shape,np.uint8)*255
    image = image0.copy() #*255.

    #image[:,:] = (0,0,0)

    kp = np.array(kp[:,:2])
    texts = []

    #colors = [
    #	(0,0,255),#red
    #	(255,255,0),#yellow
    #	(0,255,0),#green
    #	(255,0,0),#blue
    #	(75,0,130),#indigo
    #	(128,128,128),#grey
    #	(230,230,30),#mycolor6
    #	(230,30,180),#mycolor7
    #	(0,0,0)#violet
        #(0,0,0),#black
        # (255,165,0),#orange
    #]

    colors = [
        (255,0,0),#red
        (0,255,255),#yellow
        (0,255,0),#green
        (0,0,255),#blue
        (130,0,75),#indigo
        (128,128,128),#grey
        (30,230,230),#mycolor6
        (180,30,230),#mycolor7
        (0,0,0)#violet
        #(0,0,0),#black
        # (255,165,0),#orange
    ]


    if(name == 'green_hand_upper'):
        # lh_upper [5,7]
        h = int(getLength(kp[7], kp[9]) * k) #0.5
        drawLine(kp[5], kp[7], h, image, colors[2]);
        #texts.append((kp[5], "Upper left hand"))


    if(name == 'red_hand_lower'):
        # lh_bottom [7,9]
        h = int(getLength(kp[7], kp[9]) * k) #0.45
        drawLine(kp[7], kp[9], h, image, colors[0]);
        #texts.append((getCenter(kp[7],kp[9]), "Bottom left hand"))

    if(name == 'green_hand_upper'):
        # rh_upper [6,8]
        h = int(getLength(kp[8], kp[10]) * k) #0.5
        drawLine(kp[6], kp[8], h, image, colors[2]);
        #texts.append((kp[6], "Upper right hand"))

    if(name == 'red_hand_lower'):
        # rh_bottom [8,10]
        h = int(getLength(kp[8], kp[10]) * k) #0.45
        drawLine(kp[8], kp[10], h, image, colors[0]);
        #texts.append((getCenter(kp[8], kp[10]), "Bottom right hand"))

    if(name == 'gray_head'):
        #head
        h = int(getLength(kp[3],kp[4]))
        c = getCenter(kp[3],kp[4])
        p1 = rotate(c,kp[3],-math.pi/2)
        p2 = rotate(c,kp[3], math.pi/2)
        (p1,p2) = deflate(p1,p2,k)  #0.2
        drawLine(p1,p2,h,image,colors[5])
        #texts.append(( p1, "Head"))

    if(name == 'blue_body'):
        #body [5,6,11,12]
        h = int(getLength(kp[5],kp[6]))
        p1 = getCenter(kp[5],kp[6])
        p2 = getCenter(kp[11],kp[12])
        (p1,p2) = deflate(p1,p2,k) #0.4
        drawLine(p1,p2,h,image,colors[3]);
        #texts.append(( getCenter(p1,p2), "Body"))

    if(name == 'lightBlue_leg_upper'):
        #ll_upper [11,13,15]
        h = int(getLength(kp[11],kp[13])*k) #0.3
        drawLine(kp[11],kp[13],h,image,colors[6]);
        #texts.append((kp[11], "Upper left leg"))

    if(name == 'violet_leg_lower'):
        # ll_bottom [11,13,15]
        h = int(getLength(kp[13], kp[15]) * k) #0.3
        drawLine(kp[13], kp[15], h, image,colors[7]);
        #texts.append((kp[15], "Bottom left leg"))

    if(name == 'lightBlue_leg_upper'):
        # rl_upper [11,13,15]
        h = int(getLength(kp[12], kp[14]) * k) #0.3
        drawLine(kp[12], kp[14], h, image,colors[6]);
        #texts.append((kp[12], "Upper right leg"))

    if(name == 'violet_leg_lower'):
        # rl_bottom [11,13,15]
        h = int(getLength(kp[14], kp[16]) * k) #0.3
        drawLine(kp[14], kp[16], h, image,colors[7]);
        #texts.append((getCenter(kp[14],kp[16]), "Bottom right leg"))


    #image = cv2.addWeighted(image0, 0.999, image, 0.001, 0)
    #for t in texts: cv2.putText(image, t[1], tuple(t[0]),cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

    return image
def readKeypoints():
    oo = []
    for line in sys.stdin:
        o = [e.strip() for e in line.split(';')]
        oo.append({'fn': o[0], 'keypoints': json.loads(o[4])})
    return oo
def plotChart(ii,title, xlabel):
    ii = sorted(ii,key=lambda e: e['x'])
    x = list(map(lambda e: e['x'], ii))
    y = list(map(lambda e: e['y'], ii))
    plt.plot(x, y, color='green', linestyle='dashed', linewidth = 1, marker='x')
    plt.xlabel(xlabel)
    plt.ylabel('Time [s]')
    plt.title(title)
    plt.grid()
    plt.show()


def main():	
    parts = [
        {'name': 'blue_body',		'min': [0, 0, 250], 	'max': [5, 5, 255]},
        {'name': 'red_hand_lower', 	'min': [250, 0, 0], 	'max': [255, 5, 5]},
        {'name': 'green_hand_upper','min': [0, 250, 0], 	'max': [5, 255, 5]},
        {'name': 'gray_head', 		'min': [126, 126, 126], 'max': [129, 129, 129]},
        {'name': 'lightBlue_leg_upper','min': [28, 228, 228], 	'max': [32, 232, 232]},
        {'name': 'violet_leg_lower', 'min': [178, 28, 228], 	'max': [182, 32, 232]},
    ]
    for p in parts: p['kfce'] = []


    keypoints = readKeypoints()

    for k in np.arange(0, 1, 0.1):
        for p in parts: p['oo'] = []
        for kp in keypoints:
            fn = kp['fn']
            kk = np.asarray(kp['keypoints']).astype(int)
            a = readFile(fn.replace(".", "_ref."))
            b = readFile(os.path.join('ciste',fn))
            #b = drawBodyParts(kk,b,k)
            print(fn)

            for p in parts:
                c = drawBodyParts(p['name'],kk,b,k)
                res = cmp(a,c,p['min'],p['max'])
                p['oo'].append(res[2]-res[3]/1.6)

        for p in parts:
            oo=list(filter(lambda x:x!=None,p["oo"]))
            p['kfce'].append({'x':k,'y':np.mean(oo)})
            print(p['name'],k,np.mean(oo))

    for p in parts:
        plotChart(p['kfce'],p['name'],'k')



if(__name__ ==  "__main__"):
    main()    
