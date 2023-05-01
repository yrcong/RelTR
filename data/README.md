# 1. For Visual Genome:
Download **RelTR Repo** with:
```
git clone https://github.com/yrcong/RelTR.git
cd RelTR
```

## For Inference
:smile: It is super easy to configure the RelTR environment.

If you want to **infer an image**, only python=3.6, PyTorch=1.6 and matplotlib are required!
You can configure the environment as follows:
```
# create a conda environment 
conda create -n reltr python=3.6
conda activate reltr

# install packages
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
conda install matplotlib
```

## Training/Evaluation on Visual Genome
If you want to **train/evaluate** RelTR on Visual Genome, you need a little more preparation:

a) Scipy (we used 1.5.2) and pycocotools are required. 
```
conda install scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

b) Download the annotations of [Visual Genome (in COCO-format)](https://drive.google.com/file/d/1aGwEu392DiECGdvwaYr-LgqGLmWhn8yD/view?usp=sharing) and unzip it in the ```data/``` forder.

c) Download the the images of VG [Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Unzip and place all images in a folder ```data/vg/images/```

d) Some widely-used evaluation code (**IoU**) need to be compiled... We will replace it with Pytorch code.
```
# compile the code computing box intersection
cd lib/fpn
sh make.sh
```

The directory structure looks like:
```
RelTR
| 
│
└───data
│   └───vg
│       │   rel.json
│       │   test.json
│       |   train.json
|       |   val.json
|       |   images
│   └───oi
│       │   rel.json
│       │   test.json
│       |   train.json
|       |   val.json
|       |   images
└───datasets    
... 
```

# 2. Usage

## Inference
a) Download our [RelTR model](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) pretrained on the Visual Genome dataset and put it under 
```
ckpt/checkpoint0149.pth
```
b) Infer the relationships in an image with the command:
```
python inference.py --img_path $IMAGE_PATH --resume $MODEL_PATH
```
We attached 5 images from **VG** dataset and 1 image from **internet**. You can also test with your customized image. The result should look like:
<p align="center">
  <img src="demo/vg1_pred.png">
</p>

## Training
a) Train RelTR on Visual Genome on a single node with 8 GPUs (2 images per GPU):
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset vg --img_folder data/vg/images/ --ann_path data/vg/ --batch_size 2 --output_dir ckpt
```
b) Train RelTR on Open Images V6 on a single node with 8 GPUs (2 images per GPU):
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset oi --img_folder data/oi/images/ --ann_path data/oi/ --batch_size 2 --output_dir ckpt
```

## Evaluation
a) Evaluate the pretrained [RelTR](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) on Visual Genome with a single GPU (1 image per GPU):
```
python main.py --dataset vg --img_folder data/vg/images/ --ann_path data/vg/ --eval --batch_size 1 --resume ckpt/checkpoint0149.pth
```

b) Evaluate the pretrained [RelTR](https://drive.google.com/file/d/1pcoUnR0XWsvM9lJZ5f93N5TKHkLdjtnb/view?usp=share_link) on Open Images V6 with a single GPU (1 image per GPU):
```
python main.py --dataset oi --img_folder data/oi/images/ --ann_path data/oi/ --eval --batch_size 1 --resume ckpt/checkpoint0149_oi.pth
```
