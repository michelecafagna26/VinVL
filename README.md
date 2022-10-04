# VinVL
Original VinVL (and Oscar) repo with API for an easy ready-to-go inference for captioning.
All the code is on [microsoft/Oscar](https://github.com/microsoft/Oscar) please refer that repo for further info about the pre-training, fine-tuning and pretrained checkpoint.

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
git clone --recursive git@github.com:microsoft/Oscar.git
cd Oscar/coco_caption
./get_stanford_models.sh
cd ..
python setup.py build develop

# install the requirements
pip install -r requirements.txt

```

## Model downloading
We released pre-trained models, datasets, VinVL image features, and Oscar+ pretraining corpus for downstream tasks. Please check [VinVL_DOWNLOAD.md](https://github.com/microsoft/Oscar/blob/master/VinVL_DOWNLOAD.md) from the original repo for details.

To download checkpoints for the Vanilla OSCAR, please check [DOWNLOAD.md](https://github.com/microsoft/Oscar/blob/master/DOWNLOAD.md) for details.


## Quick start: captioning

```python
from Oscar.transformers.pytorch_transformers import BertConfig, BertTokenizer
from Oscar.oscar.modeling.modeling_bert import BertForImageCaptioning
from Oscar.oscar.wrappers import OscarTensorizer

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
visual_features = torch.from_numpy(feat_obj['features']).to(device).unsqueeze(0)

# labels are usually extracted by the features extractor
labels = [['boat', 'boat', 'boat', 'bottom', 'bush', 'coat', 'deck', 'deck', 'deck', 'dock', 'hair', 'jacket']]

inputs = tensorizer.encode(visual_features, labels=labels)
outputs = model(**inputs)

pred = tensorizer.decode(outputs)

# the output looks like this:
# pred = {0: [{'caption': 'a red and white boat traveling down a river next to a small boat.', 'conf': 0.7070220112800598]}
```
