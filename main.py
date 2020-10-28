from skimage import io
import matplotlib.pyplot as plt
import numpy as np

lightfield_path = "data/chessboard_lightfield.png"

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

  mosaic /= 255.0
  plt.imshow(mosaic)
  plt.show()

  return mosaic

def main():
  img_5d = load_lightfield_image()
  mosaic = create_mosaic(img_5d)

main()
