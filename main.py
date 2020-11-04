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

def refocus(img_5d, d, a):
  refocus_img = np.zeros((400,700,3))
  arange = int(2 * (a + 0.5))
  for u in range(arange + 1):
    for v in range(arange + 1):
      img = img_5d[u,v,:,:,:]
      shifted_img = shift_img(img, u, v, d)
      refocus_img += shifted_img

  # average over all images in the mosaic
  # divide by exitance
  # print(arange ** 2)
  refocus_img /= ((arange+1)**2)
  # refocus_img /= (a**2)
  # plt.imshow(refocus_img)
  # plt.show()
  return refocus_img

def AFI(img_5d):
  pu = [170, 270, 270]
  pv = [300, 300, 400]
  AFI = np.zeros((7,5,3))

  AFIs = []
  for i in range(5):
    AFIs.append(np.zeros((7,5,3)))

  for a in range(1,8):
    for f in range(5):
      rlf = refocus(img_5d, Fs[f], a)
      for i in range(len(pu)):
        (AFIs[i])[a-1][f] = rlf[pu[i]][pv[i]]

  for i in range(5):
    io.imsave("output/afi_%d_%d.png" % (pu[i], pv[i]), AFIs[i])

def af_mosaic(img_5d):
  # should result in a (6400, 11200, 3) image
  mosaic = np.zeros((2800, 3500, 3))
  for a in range(1,8):
    for f in range(5):
      rlf = refocus(img_5d, Fs[f], a)
      mosaic[(a-1)*400:a*400, f*700:(f+1)*700,:] = rlf

  io.imsave("af_mosaic.png", mosaic)

  return mosaic

def least_var_col(AFI):
  vars = np.zeros(5)
  for v in range(5):
    r_var = np.var(AFI[:,v,0])
    g_var = np.var(AFI[:,v,1])
    b_var = np.var(AFI[:,v,2])

    vars[v] = r_var + g_var + b_var

  least_var_index = np.argmin(vars)
  least_var_d = Fs[least_var_index]
  return least_var_d

def afi_depth(img_5d):
  depth_map = np.zeros((400, 700))
  mosaic = io.imread("af_mosaic.png")/255.0
  mw = mosaic.shape[1]

  for u in range(400):
    for v in range(700):
      AFI = np.zeros((7,5,3))
      for a in range(7):
        for f in range(5):
          row_i = a*400 + u
          col_i = f*700 + v
          AFI[a][f] = mosaic[row_i, col_i]

      # find column with least variance
      least_var_d = least_var_col(AFI)
      depth_map[u][v] = least_var_d

  io.imsave("afi_depth.png", depth_map)
  return depth_map

def main():
  img_5d = load_lightfield_image()

  # depth_and_allfocus()
  # af_mosaic(img_5d)
  # afi_depth(img_5d)
  AFI(img_5d)

main()
