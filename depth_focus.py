from skimage import io
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import os
from cp_hw2 import lRGB2XYZ
import cv2

ks1 = 21
ks2 = 13
sig1 = 32.0
sig2 = 32.0

def depth_and_allfocus(fstack_dir):
  # weights are the same for all of the color channels

  wsharp_all = np.zeros((400,700,3))
  img_focus = np.zeros((400,700,3))
  img_depth = np.zeros((400, 700))

  for i in range(5):
    img = io.imread(fstack_dir + ("f%d.png" % i)) # / 255.0
    img_rgb = np.where(img <= 0.0031308, 12.92*img,
                       1.055*np.power(img, (1.0/2.4)) - 0.055)
    img_xyz = lRGB2XYZ(img_rgb)
    img_lum = img_xyz[:,:,1]
    img_lfq = cv2.GaussianBlur(img_lum, (ks1, ks1), sig1)
    img_hfq = img_lum - img_lfq

    depth_i = np.ones((400, 700))
    depth_i *= (i * -0.4)

    wsharp = cv2.GaussianBlur(np.square(img_hfq), (ks2, ks2), sig2)
    wsharp_i = np.dstack((wsharp, wsharp, wsharp))
    img_focus += np.multiply(wsharp_i, img)
    img_depth += np.multiply(wsharp, depth_i)
    wsharp_all += wsharp_i

  img_focus /= wsharp_all
  img_focus = np.nan_to_num(img_focus)
  img_depth = np.divide(img_depth, wsharp_all[:,:,0])
  io.imsave("depth.png", img_depth)
  io.imsave("all_focus.png", img_focus)

depth_and_allfocus("output/focal_stack/")
