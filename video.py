from skimage import io
import time
import matplotlib.patches as patches
from skimage.morphology import disk
from scipy import interpolate
from scipy import signal
import scipy
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

def shift_img(img, sx, sy):
  # img is sub-aperture view, (400, 700, 3)
  imgw = img.shape[1]
  imgh = img.shape[0]

  x = np.arange(imgw)
  y = np.arange(imgh)

  zr = img[:,:,0]
  zg = img[:,:,1]
  zb = img[:,:,2]

  fr = interpolate.interp2d(x, y, zr)
  fg = interpolate.interp2d(x, y, zg)
  fb = interpolate.interp2d(x, y, zb)

  xmin_new = round(10 * sx) / 10.0
  xmax_new = round(10 * (imgw + sx)) / 10.0
  ymin_new = round(10 * sy) / 10.0
  ymax_new = round(10 * (imgh + sy)) / 10.0

  x_new = np.arange(xmin_new, xmax_new)
  y_new = np.arange(ymin_new, ymax_new)

  zr_new = fr(x_new, y_new)[0:imgh, 0:imgw]
  zg_new = fg(x_new, y_new)[0:imgh, 0:imgw]
  zb_new = fb(x_new, y_new)[0:imgh, 0:imgw]

  new_img = np.dstack((zr_new, zg_new, zb_new))
  return new_img


# refocus at patch g
def video_refocus(video_path, gpatch, num_frames, swu, swv, subsample):
  u0, v0, u1, v1 = gpatch
  su0 = u0 - swu
  su1 = u1 + swu
  sv0 = v0 - swv
  sv1 = v1 + swv

  pw = v1 - v0
  ph = u1 - u0

  um = u0 + (ph/2.0)
  vm = v0 + (pw/2.0)

  gs = u1 - u0

  if ((u1 - u0) != (v1 - v0)):
    print("NOT A SQUARE PATCH, RETURN")
    return

  cap = cv2.VideoCapture(video_path)

  i = 0
  focus_img = np.zeros((720, 1280, 3))
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print( length ) # 112

  gpatch = np.zeros(((u1-u0), (v1-v0)))

  cap.set(1, int(num_frames/2))
  res, mid_frame = cap.read()
  gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY) / 255.0
  mid_frame = np.divide(mid_frame, 255.0)
  gpatch = np.copy(gray[u0:u1, v0:v1])
  gpatch -= np.mean(gpatch)

  plt.imshow(mid_frame)
  plt.show()
  plt.imshow(gpatch)
  plt.show()

  cap.set(1, -1)
  i = 0
  while(i < num_frames):
    ret, frame = cap.read()
    print(i)
    if (i % subsample == 0):
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
      frame = np.divide(frame, 255.0)
      g_box = scipy.ndimage.uniform_filter(gray, size=gs)

      search_window = (gray - g_box)[su0:su1, sv0:sv1]

      ncc = signal.correlate2d(search_window, gpatch, boundary='symm', mode='same')
      sy, sx = np.unravel_index(np.argmax(ncc), ncc.shape)

      # shift frame
      shift_x = sx - swv - pw/2.0
      shift_y = sy - swu - ph/2.0

      # print(shift_x, shift_y)
      imgs = shift_img(frame, shift_x, shift_y)

      world_y = u0 + shift_y
      world_x = v0 + shift_x

    if (i % 90 == 0):
      # Create figure and axes
      fig,ax = plt.subplots(1)
      ax.imshow(frame)
      plt.plot(world_x, world_y, 'ro')
      template = patches.Rectangle((world_x,world_y),100,100,linewidth=1,edgecolor='r',facecolor='none')
      ax.add_patch(template)
      plt.show()
    focus_img += imgs
    i+=1

  # focus_img /= length
  focus_img /= (num_frames)
  focus_img *= 255
  # cv2.imwrite("media/custom_focus_mouse.png", focus_img)
  cv2.imwrite("media/custom_focus_cow.png", focus_img)
  return focus_img

def check_window(video_path, su0, sv0, su1, sv1, s):
  print(su0, su1, sv0, sv1)
  cap = cv2.VideoCapture(video_path)
  while (True):
    ret, frame = cap.read()
    start_point = (sv0 - s, su0 - s)
    end_point = (sv1 + s, su1 + s)
    color = (255, 0, 0)
    thickness = 2
    frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

    cv2.imshow("frame", frame)
    cv2.waitKey(10)


def main():
  video_path = "media/DSC_0015.MOV"
  # focus on cow
  gpatch = [350, 680, 450, 780]
  # focus on mouse
  # gpatch = [430, 180, 530, 280]

  s = 250
  # s = 180
  # check_window(video_path, gpatch[0], gpatch[1], gpatch[2], gpatch[3], s)
  # video_refocus(video_path, gpatch, 331, 50, 200, 3)
  video_refocus(video_path, gpatch, 331, s, s, 3)


main()

