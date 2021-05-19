#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""	
	Code for evaluating the probability of success of the evaluated images by the keypoint algorithm against the manually obtained groundtrue images. 
"""

import cv2
import numpy as np
from PIL import Image
import os



def cmp(a,b,cmin,cmax):
	"""
	It is assumed that images a, b have the same size and color system RGB.
	The function searches the number of identical points according to the color given by the interval cmin, cmax. 

    Args:
    a : numpy array
        referential image RGB
    b : numpy array
        evaluated image RGB
	cmin: array
		min color RGB 
	cmax: array
		max color RGB

    Returns:
    tuple 
		sum0 ... number of points of the given color in the referential image,
		sum1 ... number of points of the given color in the evaluated image,
		sum2 ... number of points of the given color simultaneously in the referential and in the evaluated image,
		sum3 ... number of points of the given color located outside the referential area,
		p1 	 ... sum2 / sum0 ratio 
    """	
	ma = (a[:,:,0] >= cmin[0]) & (a[:,:,0] <= cmax[0]) & (a[:,:,1] >= cmin[1]) & (a[:,:,1] <= cmax[1]) & (a[:,:,2] >= cmin[2]) & (a[:,:,2] <= cmax[2])  
	mb = (b[:,:,0] >= cmin[0]) & (b[:,:,0] <= cmax[0]) & (b[:,:,1] >= cmin[1]) & (b[:,:,1] <= cmax[1]) & (b[:,:,2] >= cmin[2]) & (b[:,:,2] <= cmax[2])  
		
	#Image.fromarray(a, 'RGB').show()
	#cv2.waitKey()
	
	sum0 = np.count_nonzero(ma);
	sum1 = np.count_nonzero(mb);
	sum2 = np.count_nonzero(np.logical_and(ma,mb));	
	sum3 = np.count_nonzero(np.logical_xor(ma,mb));	
	p1 = sum2/sum0;
	p2 = sum3/sum0;
	return (sum0,sum1,sum2,sum3,p1)

def readFile(name):
	"""
	Read file by name from disk, converts it from BGR to RGB color system and returns image as numpy array 

    Args:
    name : str
        file name
    
    Returns:
    numpy array:
        loaded image as a numpy array
    """
	im = cv2.imread(name)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	return np.asarray(im)
	
def main():
	"""
	Main procedure. All images with postfix _bp, which were evaluated by the keypoint algorithm, are gradually loaded from the current directory.
	For each evaluated image, its reference equivalent with postfix _ref is loaded, which was evaluated manually in photoshop. 
	Both images are compared according to all searched body parts. The results of the comparison are sent to standard output and at the same time 
	added to each body part. After passing through all the images, the arithnetic mean of the results for each body part is performed and the summary 
	results are written to standard output. 	
	"""
	parts = [
        {'name': 'blue_body',		'min': [0, 0, 250], 	'max': [5, 5, 255]},
		{'name': 'red_hand_down', 	'min': [250, 0, 0], 	'max': [255, 5, 5]},
		{'name': 'green_hand_up', 	'min': [0, 250, 0], 	'max': [5, 255, 5]},
		{'name': 'gray_head', 		'min': [126, 126, 126], 'max': [129, 129, 129]},
		{'name': 'lightBlue_leg_up','min': [28, 228, 228], 	'max': [32, 232, 232]},
		{'name': 'violet_leg_down', 'min': [178, 28, 228], 	'max': [182, 32, 232]},
	]
	for p in parts: p['oo'] = []
	
	def flt(f): 
		return f.lower().endswith(('_bp.png', '_bp.jpg', '_bp.jpeg'))  and  '_ref.' not in f.lower()
	
	for fn in filter(flt, os.listdir(".")):
		a = readFile(fn.replace("_bp.", "_ref."))	
		b = readFile(fn)	
		#print(fn,np.shape(a),np.shape(b))	
		for p in parts:
			res = cmp(a,b,p['min'],p['max'])
			p['oo'].append(res[4])
			print(fn,p['name'],res)		
	
	for p in parts:	print(p['name'],'\t',np.mean(p['oo']))
	
if(__name__ ==  "__main__"):
    main()    


