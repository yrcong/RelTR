# RelTR: RelTR: Relation Transformer for Scene Graph Generation

PyTorch Implementation of the Paper **RelTR: Relation Transformer for Scene Graph Generation**

Different from most existing advanced approaches that infer the **dense** relationships between all entity proposals, our one-stage method can directly generate a **sparse** scene graph by decoding the visual appearance.

<p align="center">
  <img src="demo/demo.png">
</p>

# Checklist

- [x] Inference Code :tada:
- [ ] Training Code for Visual Genome :clock9:
- [ ] Evaluation Code for Visual Genome :clock9:
- [ ] Training Code for OpenImages V6 :clock9:
- [ ] Evaluation Code for OpenImages V6 :clock9:

# Installation
:smile: It is super easy to configure the RelTR environment.

Only python=3.6, PyTorch=1.6 and matplotlib are required to infer an image!
You can configure the environment as follows:
```
# create a conda environment 
conda create -n reltr python=3.6
conda activate reltr

# install packages
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
conda install matplotlib
```

# Usage

## Inference
1. Download our [RelTR model](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) pretrained on the Visual Genome dataset and put it under 
```
ckpt/checkpoint0149.pth
```
2. Infer the relationships in an image with the command:
```
python inference.py --img_path $IMAGE_PATH --resume $MODEL_PATH
```
We attached 5 images from **VG** dataset and 1 image from **internet**. You can also test with your customized image. The result should look like:
<p align="center">
  <img src="demo/vg1_pred.png">
</p>

