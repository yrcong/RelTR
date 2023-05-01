# For Visual Genome:
a) Download the the images of Visual Genome [Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Unzip and place all images in a folder ```data/vg/images/```

b) Download the annotations of [VG (in COCO-format)](https://drive.google.com/file/d/1aGwEu392DiECGdvwaYr-LgqGLmWhn8yD/view?usp=sharing) and unzip it in the ```data/``` forder.


# For Open Images V6:
a) Download the original annotation of Open Images V6 (oidv6/v4-train/test/validation-annotations-vrd.csv) from the offical [website](https://storage.googleapis.com/openimages/web/download.html).

b) Download the OIv6 images provided by [Rongjie Li](https://github.com/SHTUPLUS/PySGG/blob/main/DATASET.md) and unzip it.

c) Change the paths in [process.py](https://github.com/yrcong/RelTR/blob/main/data/process.py). The images will be renamed (since the COCO tool only supports numeric image names) and the COCO-like annotations will be produced. You can also download the annotations with [the link](https://drive.google.com/file/d/1kWeG3O071Bx17KI7oLbMdgGvE5xmyY8k/view?usp=share_link), but still need to rename the images.

d) Put the renamed images in the ```data/oi/images/``` and the annotations in ```data/oi/```.
