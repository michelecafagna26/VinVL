# VinVL for Captioning
VinVL (and Oscar) implementation with APIs for an easy ready-to-go inference for captioning.
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
python apex/setup.py install --cuda_ext --cpp_ext

# install oscar
git clone --recursive git@github.com:michelecafagna26/VinVL.git
./VinVL/Oscar/coco_caption/get_stanford_models.sh
python VinVL/Oscar/setup.py build develop

# install the requirements
pip install -r VinVL/Oscar/requirements.txt

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

#ckpt = "vinvl-base-finetuned-hl-scenes-image-captioning" # if you downloaded from huggingface
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
@inproceedings{cafagna-etal-2022-understanding,
    title = "Understanding Cross-modal Interactions in {V}{\&}{L} Models that Generate Scene Descriptions",
    author = "Cafagna, Michele  and
      Deemter, Kees van  and
      Gatt, Albert",
    booktitle = "Proceedings of the Workshop on Unimodal and Multimodal Induction of Linguistic Structures (UM-IoS)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.umios-1.6",
    pages = "56--72",
    abstract = "Image captioning models tend to describe images in an object-centric way, emphasising visible objects. But image descriptions can also abstract away from objects and describe the type of scene depicted. In this paper, we explore the potential of a state of the art Vision and Language model, VinVL, to caption images at the scene level using (1) a novel dataset which pairs images with both object-centric and scene descriptions. Through (2) an in-depth analysis of the effect of the fine-tuning, we show (3) that a small amount of curated data suffices to generate scene descriptions without losing the capability to identify object-level concepts in the scene; the model acquires a more holistic view of the image compared to when object-centric descriptions are generated. We discuss the parallels between these results and insights from computational and cognitive science research on scene perception.",
}

```

## License
Oscar is released under the MIT license. See [LICENSE](LICENSE) for details. 
