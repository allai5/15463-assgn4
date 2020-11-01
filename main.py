from skimage import io
from skimage.morphology import disk
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math

lightfield_path = "data/chessboard_lightfield.png"
mosaic_path = "output/mosaic.png"
# f0 is the image with focus at the top
f0_path = "output/f0.png"
f1_path = "output/f1.png"
f2_path = "output/f2.png"
f3_path = "output/f3.png"
f4_path = "output/f4.png"

# code taken from writeup
lensletSize = 16
maxUV = (lensletSize - 1) / 2
u_ctrs = np.arange(lensletSize) - maxUV
v_ctrs = np.arange(lensletSize) - maxUV

Fs = np.arange(-1.6, 0.4, 0.4)
As = np.arange(1.0, 9.0, 1.0)

def load_lightfield_image():
  img_raw = io.imread(lightfield_path)/255.0
  lf = []
  for s in range(16):
    ls = []
    for t in range(16):
      view = img_raw[s::16,t::16,:]
      ls.append(view)
    lf.append(ls)
  img_5d = np.array(lf)
  return img_5d

def create_mosaic(img_5d):
  # should result in a (6400, 11200, 3) image
  mosaic = np.zeros((6400, 11200, 3))

  for u in range(16):
    for v in range(16):
      view = img_5d[u,v,:,:,:]
      mosaic[u*400:(u+1)*400, v*700:(v+1)*700,:] = view

  # mosaic /= 255.0
  return mosaic

def shift_img(img, u, v, d):
  # img is sub-aperture view, (400, 700, 3)
  if (d == 0): return img
  u_ctr = u_ctrs[u]
  v_ctr = v_ctrs[v]
  du = d*u_ctr
  dv = d*v_ctr

  x = np.arange(700*v, 700*(v+1))
  y = np.arange(400*u, 400*(u+1))
  zr = img[:,:,0]
  zg = img[:,:,1]
  zb = img[:,:,2]

  fr = interpolate.interp2d(x, y, zr)
  fg = interpolate.interp2d(x, y, zg)
  fb = interpolate.interp2d(x, y, zb)

  xmin_new = round(10*(700*v + dv)) / 10.0
  xmax_new = round(10*(700*(v+1) + dv)) / 10.0
  ymin_new = round(10*(400*u - du)) / 10.0
  ymax_new = round(10*(400*(u+1) - du)) / 10.0

  x_new = np.arange(xmin_new, xmax_new)
  y_new = np.arange(ymin_new, ymax_new)

  zr_new = fr(x_new, y_new)[0:400, 0:700]
  zg_new = fg(x_new, y_new)[0:400, 0:700]
  zb_new = fb(x_new, y_new)[0:400, 0:700]

  new_img = np.dstack((zr_new, zg_new, zb_new))
  return new_img


def blur_lightfield(img_5d, asize):
  img_disk = disk(asize)
  blur_img_5d = np.zeros((img_5d.shape))
  area = math.pi * math.pow(asize, 2)

  for u in range(16):
    for v in range(16):
      # original_img = img_5d[u,v,:,:,:]
      for i in range(3):
        # original_img = img_5d[u,v,:,:,i].copy()
        # imgc = img_5d[u,v,:,:,i].copy()
        imgc = img_5d[u,v,:,:,i]
        blur_img_5d[u,v,:,:,i] = signal.convolve2d(imgc, img_disk, mode='same')/area
        # diff_img = original_img - new_img

      # new_img = blur_img_5d[u,v,:,:,:]
      # diff_img = original_img - new_img
      # fig = plt.figure()
      # fig.add_subplot(2,1,1)
      # plt.imshow(original_img)
      # fig.add_subplot(2,1,2)
      # plt.imshow(new_img)
      # plt.show()

  return blur_img_5d

def refocus(img_5d, d=0):
  refocus_img = np.zeros((400,700,3))

  for u in range(16):
    for v in range(16):
      img = img_5d[u,v,:,:,:]
      shifted_img = shift_img(img, u, v, d)
      refocus_img += shifted_img

  # average over all 256 images in the mosaic
  refocus_img /= 256.0
  # refocus_img /= 255.0

  # normalized
  return refocus_img

def AFI(img_5d, pu, pv):
  AFI = np.zeros((7,5, 3))
  for a in range(7):
    lf = blur_lightfield(img_5d, As[a])
    for f in range(5):
      print("refocus")
      rlf = refocus(lf, Fs[f])
      AFI[a][f] = rlf[pu][pv]


  plt.imshow(AFI)
  plt.show()
  return AFI

def main():
  img_5d = load_lightfield_image()
  # depth_and_allfocus()
  AFI(img_5d, 270, 500)

main()
