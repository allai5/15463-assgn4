from skimage import io
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

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

def load_lightfield_image():
  img_raw = io.imread(lightfield_path)
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

def refocus(img_5d, d=0):
  refocus_img = np.zeros((400,700,3))

  for u in range(16):
    for v in range(16):
      img = img_5d[u,v,:,:,:]
      shifted_img = shift_img(img, u, v, d)
      refocus_img += shifted_img

  # average over all 256 images in the mosaic
  refocus_img /= 256.0
  refocus_img /= 255.0

  # normalized
  return refocus_img

def main():
  img_5d = load_lightfield_image()
  depth_and_allfocus()

main()
