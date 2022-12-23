# VinVL for Captioning
Original VinVL (and Oscar) implementation with APIs for an easy ready-to-go inference for captioning.
The code is based on [microsoft/Oscar](https://github.com/microsoft/Oscar) please refer that repo for further info about the pre-training, fine-tuning and pretrained checkpoint.

## Installation
Create your virtual environment an install the following dependencies according to your system specs.
### Requirements
- Python 3.7
- Pytorch
- torchvision
- cuda

Then run:
```bash
# good practice
pip install --upgrade pip

# install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install oscar
cd ..
git clone --recursive git@github.com:michelecafagna26/VinVL.git
cd Oscar/coco_caption
./get_stanford_models.sh
cd ..
python setup.py build develop

# install the requirements
pip install -r requirements.txt

```

## Model downloading 
Please check [VinVL_DOWNLOAD.md](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md) from the original repo for details regarding pre-trained models, datasets, VinVL image features, and Oscar+ pretraining corpus for downstream tasks 

To download checkpoints for the Vanilla OSCAR, please check [DOWNLOAD.md](https://github.com/microsoft/Oscar/blob/master/DOWNLOAD.md) for details.

## Model downloading (Huggingface)

- VinVL base pretrained for image captioning

```bash
# download from the huggingface model hub
git lfs install # if not installed
git clone https://huggingface.co/michelecafagna26/vinvl-base-image-captioning
```
- VinVL base finetuned for scene description generation

```bash
# download from the huggingface model hub
git lfs install # if not installed
git clone https://huggingface.co/michelecafagna26/vinvl-base-finetuned-hl-scenes-image-captioning
```

## Features Extraction

For an easy feature extraction I've prepared a repo based on the VinVL's original VisualBackbone: [michelecafagna26/vinvl-visualbackbone](https://github.com/michelecafagna26/vinvl-visualbackbone)

---
## Quick start: Image Captioning

```python
from transformers.pytorch_transformers import BertConfig, BertTokenizer
from oscar.modeling.modeling_bert import BertForImageCaptioning
from oscar.wrappers import OscarTensorizer

ckpt = "path/to/the/checkpoint"
device = "cuda" if torch.cuda.is_available() else "cpu"

# original code
config = BertConfig.from_pretrained(ckpt)
tokenizer = BertTokenizer.from_pretrained(ckpt)
model = BertForImageCaptioning.from_pretrained(ckpt, config=config).to(device)

# This takes care of the preprocessing
tensorizer = OscarTensorizer(tokenizer=tokenizer, device=device)

# numpy-arrays with shape (1, num_boxes, feat_size)
# feat_size is 2054 by default in VinVL
visual_features = torch.from_numpy(feat_obj).to(device).unsqueeze(0)

# labels are usually extracted by the features extractor
labels = [['boat', 'boat', 'boat', 'bottom', 'bush', 'coat', 'deck', 'deck', 'deck', 'dock', 'hair', 'jacket']]

inputs = tensorizer.encode(visual_features, labels=labels)
outputs = model(**inputs)

pred = tensorizer.decode(outputs)

# the output looks like this:
# pred = {0: [{'caption': 'a red and white boat traveling down a river next to a small boat.', 'conf': 0.7070220112800598]}
```
## Demo
Coming Soon!

## Citations
Please consider citing the original papers if you use the code:
```BibTeX
@article{li2020oscar,
  title={Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks},
  author={Li, Xiujun and Yin, Xi and Li, Chunyuan and Hu, Xiaowei and Zhang, Pengchuan and Zhang, Lei and Wang, Lijuan and Hu, Houdong and Dong, Li and Wei, Furu and Choi, Yejin and Gao, Jianfeng},
  journal={ECCV 2020},
  year={2020}
}

@article{zhang2021vinvl,
  title={VinVL: Making Visual Representations Matter in Vision-Language Models},
  author={Zhang, Pengchuan and Li, Xiujun and Hu, Xiaowei and Yang, Jianwei and Zhang, Lei and Wang, Lijuan and Choi, Yejin and Gao, Jianfeng},
  journal={CVPR 2021},
  year={2021}
}
```

For the model finetuned on scene descriptions
```BibTeX
@misc{cafagna2022understanding,
  author = {Cafagna Michele and van Deemter Kees and Gatt Albert},
  title = {Understanding Cross-modal Interactions in V&amp;L Models that Generate Scene Descriptions},
  doi = {10.48550/ARXIV.2211.04971},
  url = {https://arxiv.org/abs/2211.04971},
  keywords = {Computation and Language (cs.CL), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## License
Oscar is released under the MIT license. See [LICENSE](LICENSE) for details. 
