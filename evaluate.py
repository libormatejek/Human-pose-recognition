#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 09:22:33 2020

@author: libor

This module contains code for classification objects in the image. It can identify the contained objects, highlight them with 
a transparent mask and annotate them. If people appear in the pictures, it is possible also recognize parts of people's bodies 
such as leg, arm, torso and head. It even recognizes the upper and lower limbs. For evaluation are used neural networks 
maskrcnn_resnet50_fpn and keypointrcnn_resnet50_fpn trained using a coco dataset. 
"""

import os
import matplotlib
import torch
import torchvision
import numpy as np
import cv2
import argparse
from PIL import Image
from torchvision.transforms import transforms as transforms
import random
import matplotlib.pyplot as plt
import math
import time
import json

class TaskMask(object):
    """
    The input image is converted to a numpy array then to a tensor, classification is performed using the maskrcnn_resnet50_fpn model and mask calculation,
    the probability distribution of classified objects is processed, objects with a probability of at least 0.5 are selected. Next, it goes through the
    individual masks and ensure their graphical highlighting by modifying the color of the point. Output image containing masks and the smallest enveloping
    rectangles of found objects is displayed according to settings and is always saved to disk.
    """
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.transform = transforms.Compose([transforms.ToTensor()])
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.model.to(self.device)
        self.model.eval()

    def segmentation(self,inputPath, outputPath, show, threshold=0.5, rect_th=1, text_size=2, text_th=1):
        """
        Converts input image to output image. Output image contains highlited masks and annotations of classified objects
        Args:
            inputPath: str
                path to input file
            outputPath: str
                path to output file
            show: bool
                display image immediately after processing
            threshold: float
                threshold for probability of object clasification
            rect_th: int
                rectangle line thickness
            text_size: int
                font size of annotation text
            text_th: int
                annotation text thickness
        Returns:
            list
                names of classified objects in input image
        """
        masks, boxes, pred_cls = self.get_prediction(inputPath, threshold)
        img = cv2.imread(inputPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(masks)):
            rgb_mask = self.random_colour_masks(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        if show:
            plt.figure(figsize=(10,10))
            plt.imshow(img) #plt.xticks([]) plt.yticks([])
            plt.show()
        print(outputPath)
        cv2.imwrite(outputPath, img)
        return pred_cls

    def get_prediction(self,img_path, threshold):
        img = Image.open(img_path)
        img = self.transform(img)
        #img = img.to(self.device)
        pred = self.model([img])
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        pred_class = [TaskMask.COCO_CLASSES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return masks, pred_boxes, pred_class

    def random_colour_masks(self,image):
        colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask



def forEach(list, fce):
    for i, v in enumerate(list):
        fce(v, i, list)

def getLength(p1,p2):
    return pow(pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2),0.5)

def getCenter(p1,p2):
    return ((p1+p2)/2).astype("i")
    #return [int((p1[0] + p2[0])/2),int((p1[1] + p2[1])/2)]

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


class TaskKp(object):
    """
    The input image is converted to a numpy array then to a tensor, the input vector thus created enters the model keypointrcnn_resnet50_fpn,
    which returns found objects of type (score, keypoints). Score means the probability that it is a person and only objects are selected with
    a probability > 0.9. For such selected objects, keypoints are drawn in the original image. Keypoint interconnection is statically defined,
    the predictor returns only the coordinates of the points in the defined order. The resulting image is optionally displayed during processing 
    and is always saved at the same time to the output directory.
    """
    # pairs of edges for 17 of the keypoints detected ...
    # ... these show which points to be connected to which point ...
    # ... we can omit any of the connecting points if we want, basically ...
    # ... we can easily connect less than or equal to 17 pairs of points ...
    # ... for keypoint RCNN, not  mandatory to join all 17 keypoint pairs
    EDGES = [(0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),(5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),(12, 14), (14, 16), (5, 6)]

    def __init__(self):
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,num_keypoints=17)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def segmentation(self,inputPath,kpOutputPath,bpOutputPath,show):
        """
        Converts input image to output image. Output image contains highlited bodyparts of persons and its annotations

        Args:
            inputPath: str
                path to input file
            kpOutputPath: str
                path to output file with keypoints
            bpOutputPath: str
                path to output file with highlited bodyparts
            show: bool
                display image immediately after processing
        """
        dt = time.time()
        imageNp,image = self.imageToInputVector(inputPath)
        image = image.to(self.device)
        with torch.no_grad(): outputs = self.model(image)
        dt = time.time() - dt
        kpImage = self.draw_keypoints(outputs, imageNp.copy())
        bpImage,kk = self.draw_bodyparts(outputs, imageNp.copy())
        if show:
            plt.figure(figsize=(10,10))
            plt.imshow(kpImage)#plt.xticks([])plt.yticks([])
            plt.show()
            plt.figure(figsize=(10,10))
            plt.imshow(bpImage)#plt.xticks([])plt.yticks([])
            plt.show()
        #print(kpOutputPath)
        #cv2.imwrite(kpOutputPath, kpImage*255.)
        print(bpOutputPath)
        cv2.imwrite(bpOutputPath, bpImage)
        return dt,kk

    def imageToInputVector(self,imagePath):
        image = Image.open(imagePath).convert('RGB')
        imageNp = np.array(image, dtype=np.float32)
        imageNp = cv2.cvtColor(imageNp, cv2.COLOR_RGB2BGR) / 255.
        image = self.transform(image)
        image = image.unsqueeze(0)
        return (imageNp,image)



    def drawBodyParts(self,kp,image0):

        #image = np.ones(image0.shape,np.uint8)*255
        image = image0.copy()*255.
        #image[:,:] = (0,0,0)

        kp = np.array(kp[:,:2])
        texts = []

        colors = [
            (0,0,255),#red
            (255,255,0),#yellow
            (0,255,0),#green
            (255,0,0),#blue
            (75,0,130),#indigo
            (128,128,128),#grey
            (230,230,30),#mycolor6
            (230,30,180),#mycolor7
            (0,0,0)#violet
            #(0,0,0),#black
            # (255,165,0),#orange
        ]

        #lh
        #h = int(getLength(kp[7],kp[9]) * 0.4)
        #drawLine(kp[5],kp[7],h,image,colors[0])
        #drawLine(kp[7],kp[9],h,image,colors[0])
        #texts.append(( kp[7], "Left hand"))

        # lh_upper [5,7]
        h = int(getLength(kp[7], kp[9]) * 0.5)
        drawLine(kp[5], kp[7], h, image, colors[2]);
        #texts.append((kp[5], "Upper left hand"))

        # lh_bottom [7,9]
        h = int(getLength(kp[7], kp[9]) * 0.45)
        drawLine(kp[7], kp[9], h, image, colors[0]);
        #texts.append((getCenter(kp[7],kp[9]), "Bottom left hand"))


        #rh
        #h = int(getLength(kp[8],kp[10]) * 0.4)
        #drawLine(kp[6],kp[8],h,image,colors[0])
        #drawLine(kp[8],kp[10],h,image,colors[0])
        #texts.append(( kp[8], "Right hand"))

        # rh_upper [6,8]
        h = int(getLength(kp[8], kp[10]) * 0.5)
        drawLine(kp[6], kp[8], h, image, colors[2]);
        #texts.append((kp[6], "Upper right hand"))

        # rh_bottom [8,10]
        h = int(getLength(kp[8], kp[10]) * 0.45)
        drawLine(kp[8], kp[10], h, image, colors[0]);
        #texts.append((getCenter(kp[8], kp[10]), "Bottom right hand"))

        #head
        h = int(getLength(kp[3],kp[4]))
        c = getCenter(kp[3],kp[4])
        p1 = rotate(c,kp[3],-math.pi/2)
        p2 = rotate(c,kp[3], math.pi/2)
        (p1,p2) = deflate(p1,p2,0.2)
        drawLine(p1,p2,h,image,colors[5])
        #texts.append(( p1, "Head"))

        #body [5,6,11,12]
        h = int(getLength(kp[5],kp[6]))
        p1 = getCenter(kp[5],kp[6])
        p2 = getCenter(kp[11],kp[12])
        (p1,p2) = deflate(p1,p2,0.4)
        drawLine(p1,p2,h,image,colors[3]);
        #texts.append(( getCenter(p1,p2), "Body"))

        #ll_upper [11,13,15]
        h = int(getLength(kp[11],kp[13])*0.3)
        drawLine(kp[11],kp[13],h,image,colors[6]);
        #texts.append((kp[11], "Upper left leg"))

        # ll_bottom [11,13,15]
        h = int(getLength(kp[11], kp[13]) * 0.3)
        drawLine(kp[13], kp[15], h, image,colors[7]);
        #texts.append((kp[15], "Bottom left leg"))

        # rl_upper [11,13,15]
        h = int(getLength(kp[12], kp[14]) * 0.3)
        drawLine(kp[12], kp[14], h, image,colors[6]);
        #texts.append((kp[12], "Upper right leg"))

        # rl_bottom [11,13,15]
        h = int(getLength(kp[12], kp[14]) * 0.3)
        drawLine(kp[14], kp[16], h, image,colors[7]);
        #texts.append((getCenter(kp[14],kp[16]), "Bottom right leg"))


        #image = cv2.addWeighted(image0, 0.999, image, 0.001, 0)
        for t in texts:
            cv2.putText(image, t[1], tuple(t[0]),cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

        return image


    def draw_keypoints(self,outputs, image):
        # the `outputs` is list which in-turn contains the dictionaries 
        for i in range(len(outputs[0]['keypoints'])):
            keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
            # proceed to draw the lines if the confidence score is above 0.9

            if outputs[0]['scores'][i] > 0.9:
                keypoints = keypoints[:, :].reshape(-1, 3)
                for p in range(keypoints.shape[0]):
                    # draw the keypoints
                    cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    # uncomment the following lines if you want to put keypoint number
                    cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                for ie, e in enumerate(TaskKp.EDGES):
                    # get different colors for the edgkeys
                    rgb = matplotlib.colors.hsv_to_rgb([
                        ie/float(len(TaskKp.EDGES)), 1.0, 1.0
                    ])
                    rgb = rgb*255
                    # join the keypoint pairs to draw the skeletal structure
                    cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]), (keypoints[e, 0][1], keypoints[e, 1][1]), tuple(rgb), 2, lineType=cv2.LINE_AA)
            else:
                continue
        return image

    def draw_bodyparts(self,outputs, image):
        # the `outputs` is list which in-turn contains the dictionaries 
        for i in range(len(outputs[0]['keypoints'])):
            keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
            # proceed to draw the lines if the confidence score is above 0.9

            if outputs[0]['scores'][i] > 0.9:
                keypoints = keypoints[:, :].reshape(-1, 3)
                for p in range(keypoints.shape[0]):
                    # draw the keypoints
                    cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    # uncomment the following lines if you want to put keypoint number
                    #cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                image = self.drawBodyParts(keypoints,image)
                kk = keypoints
            else:
                continue
        return image,kk

def parseCmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=False, default=".", help='path to the input directory')
    parser.add_argument('-o', '--output', required=False, default=".", help='path to the output directory')
    parser.add_argument('-s', '--show', required=False, default=False, help='shows every processed image and user confirms continue by pressing key')
    args = vars(parser.parse_args())
    args = type('',(object,),args)
    args.show = args.show.lower() in ["true", "t", "yes", "y", "1"]
    #args = {'input': '.','output':'.', 'show': False}
    return args

def plotChart(ii):
    ii = sorted(ii,key=lambda e: e['x'])
    x = list(map(lambda e: e['x'], ii))
    y = list(map(lambda e: e['y'], ii))
    plt.plot(x, y, color='green', linestyle='dashed', linewidth = 1, marker='x')
    plt.xlabel('Size [1000 pixels]')
    plt.ylabel('Time [s]')
    plt.title('Závislost rychlosti zpracování na velikosti obrázku')
    plt.grid()
    plt.show()

def main():
    """
    Instances of the two base tasks TaskMask and TaskKp are created. Subsequently, the directory of input files is browsed,
    they are selected only jpg, png, jpeg images and not already segmented files containing postfix _mask or _kp. Both tasks
    load images, processes and saves them to the output directory or displays them to user immediately after processing. If
    TaskMask finds at least one person also starts searching for keypoints and then evaluates the bodyparts.
    """
    args = parseCmdline()
    taskMask = TaskMask()
    taskKp = TaskKp()
    oo = [];

    for fn in filter(lambda f: f.lower().endswith(('.png', '.jpg', '.jpeg'))  and  '_mask.' not in f.lower() and  '_kp.'   not in f.lower() and  '_bp.'   not in f.lower(), os.listdir(args.input)) :
        inputFile = os.path.join(args.input,fn)
        #classes = taskMask.segmentation(inputFile, os.path.join(args.output, fn.replace(".", "_mask.")), args.show)
        #if("person" in classes): 
        dt,kk = taskKp.segmentation(inputFile, os.path.join(args.output, fn.replace(".", "_kp.")),os.path.join(args.output, fn.replace(".", "_bp.")), args.show )
        w,h = Image.open(inputFile).size
        print(fn,dt,w,h,json.dumps((kk.tolist())))
        oo.append({'x':w*h,'y':dt})

    plotChart(oo)

if(__name__ ==  "__main__"):
    main()

