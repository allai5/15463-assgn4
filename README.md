## 1. Lightfield rendering, focal stacks, and depth from focus

### Initials

    - main.py, L26

### Sub-Aperture Views Mosaic

    - main.py, L38

#### Image Files

    - Mosaic of Sub-Aperture Views : media/mosaic.png

### Refocusing and focal-stack generation

    - main.py, L84

#### Image Files

    - Focal Stack Directory: media/focal_stack/
    - d0 corresponds to d = 0, d1.png -> d = -0.4, etc.

### All-focus image and depth from focus

    - depth_focus.py, L14

#### Image Files

    - all-focus image: media/all_focus.png
    - depth from focus: media/depth.png


### Focal-aperture stack and confocal stereo

    - generate depth map from confocal stereo: main.py, L145
    - generate aperture stack mosaic: main.py, L120

#### Image Files

    - AFI Images (7 apertures x 5 focal depths): media/afi_imgs/
    - depth from confocal stereo: media/afi_imgs/afi_depth.png
    - 2D collage of aperture stack: media/afi_imgs/afi_mosaic.png


## 3. Capture and refocus your own lightfield

    - video.py, L144

#### Image/Video Files

    - Focus on cow: media/custom_focus_cow.png
    - Focus on mouse: media/custom_focus_mouse.png
    - Video File: media/DSC_0015.MOV
