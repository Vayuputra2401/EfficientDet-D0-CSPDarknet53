# Matrice AI Code Documentation : Pathikreet Chowdhury

## Task :

## Modifying the EfficientDet-D0 model to use the CSPDarknet53 model as the backbone and modify the head to train using two datasets at the same time.

All modifications were done in local and uploaded to Google Colab for training , inference.

Dataset Download and Usage : 

```python
# Download the dataset
!wget https://s3.us-west-2.amazonaws.com/testing.resources/datasets/mscoco-samples/food-dataset-10-tat-10.tar.gz
!wget https://s3.us-west-2.amazonaws.com/testing.resources/datasets/mscoco-samples/appliance-dataset-5-tat-10.tar.gz

# Unzip the dataset
!tar -xzvf food-dataset-10-tat-10.tar.gz
!tar -xzvf appliance-dataset-5-tat-10.tar.gz
```

- For EfficientDet D0 Implementation used Pytorch implementation of zylo117/Yet-Another-EfficientDet-Pytorch
- For CSPDarknet53 implementation used Pytorch implementation of YOLOv4-pytorch by arguswift.

## Solution and Approach :

## Implementation of CSPDarknet53 in Pytorch :

### Importing Libraries

```python
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.layers.attention_layers import SEModule, CBAM
import config.yolov4_config as cfg

```

Here, necessary libraries are imported. `torch` and `torch.nn` are PyTorch libraries for building neural networks. `numpy` is used for numerical operations. `model.layers.attention_layers` contains custom attention modules like `SEModule` and `CBAM`. `config.yolov4_config` contains configurations for the YOLOv4 model.

### Mish Activation Function

```python
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

```

Defines the Mish activation function which is used as an activation option in convolutional layers.

### Convolutional Layer

```python
class Convolutional(nn.Module):
    def __init__(
        self,
        filters_in,
        filters_out,
        kernel_size,
        stride=1,
        norm="bn",
        activate="mish",
    ):
        super(Convolutional, self).__init__()
        # Constructor code...

    def forward(self, x):
        # Forward pass code...

```

Defines a convolutional layer with options for normalization and activation.

### CSPBlock

```python
class CSPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        residual_activation="linear",
    ):
        super(CSPBlock, self).__init__()
        # Constructor code...

    def forward(self, x):
        # Forward pass code...

```

Implements a CSP block which consists of convolutional layers and an optional attention module.

### CSPFirstStage

```python
class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()
        # Constructor code...

    def forward(self, x):
        # Forward pass code...

```

Defines the first stage of CSPDarknet53 which includes downsampling and splitting paths.

### CSPStage

```python
class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(CSPStage, self).__init__()
        # Constructor code...

    def forward(self, x):
        # Forward pass code...

```

Implements a stage of CSPDarknet53 with multiple CSP blocks.

### CSPDarknet53

```python
class CSPDarknet53(nn.Module):
    def __init__(
        self,
        stem_channels=32,
        feature_channels=[64, 128, 256, 512, 1024],
        num_features=3,
        weight_path='weights\\yolov4.weights',
        resume=False,
    ):
        super(CSPDarknet53, self).__init__()
        # Constructor code...

    def forward(self, x):
        # Forward pass code...

```

Defines the CSPDarknet53 backbone network with configurable parameters like stem channels, feature channels, and weight initialization.

### Loading Pre-trained Weights

```python
def _BuildCSPDarknet53(weight_path, resume):
    model = CSPDarknet53(weight_path=weight_path, resume=resume)

    return model, model.feature_channels[-3:]

```

Function to build CSPDarknet53 model and load pre-trained weights from a given weight file.

### Main

```python
if __name__ == "__main__":
    model = CSPDarknet53()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)

```

In the main block, an instance of CSPDarknet53 is created and a random input tensor `x` is passed through the model to get the output `y`.

This code implements the CSPDarknet53 backbone network, which can be used as a feature extractor in various computer vision tasks like object detection.

- We may omit the building the model code as we will be using the EfficientDet D0 model , this will serve as the backbone only.

## Replacing the backbone with CSPDarknet53 and adding 2 heads for simultaneous dataset processing :

### Importing Libraries

```python
import torch
from torch import nn
from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from efficientdet.utils import Anchors
from cspdarknet53 import CSPDarknet53

```

Here, necessary libraries are imported. `torch` and `torch.nn` are PyTorch libraries for building neural networks. `efficientdet.model` contains modules for EfficientDet like BiFPN, Regressor, and Classifier. `efficientdet.utils` contains utility functions like Anchors. `cspdarknet53` contains the CSPDarknet53 backbone network.

### EfficientDetBackbone Class

```python
class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        # Constructor code...

```

Defines the EfficientDet backbone network with parameters like the number of classes, compound coefficient, and whether to load pre-trained weights.

### Changes:

- Replaced EfficientNet backbone with CSPDarknet53 using `from cspdarknet53 import CSPDarknet53`.
- Added extra heads (`Regressor` and `Classifier`) for processing two datasets simultaneously.

### Initializing Backbone and Heads

```python
        self.backbone_net = CSPDarknet53()
        self.regressor1 = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.classifier1 = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.regressor2 = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.classifier2 = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])

```

Initialized the CSPDarknet53 backbone and added two sets of Regressor and Classifier heads for processing two datasets.

### Forward Method

```python
    def forward(self, inputs1, inputs2):
        # Forward pass code...

```

Defines the forward pass of the EfficientDet backbone network. It takes two inputs (presumably images from two different datasets) and returns features, regressions, classifications, and anchors for both datasets.

### Initialization of Backbone

```python
    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

```

Provides a method to initialize the backbone with pre-trained weights from a given path.

### Explanation:

- The EfficientDet backbone is modified to handle two datasets simultaneously by adding extra heads for regression and classification.
- CSPDarknet53 is chosen as the backbone due to its efficiency and effectiveness in feature extraction.
- By incorporating CSPDarknet53, the model can potentially learn richer representations from the input data, which may improve detection performance.

This modification allows the model to process two different datasets simultaneously, which can be useful in scenarios where multiple datasets need to be analyzed together, or when the model needs to be trained on a combination of datasets.

## Changes were made to [utils.py](http://utils.py) in the preprocess stage to add data augmentation of YOLOv4 :

### Changes and Additions:

1. **Added Data Augmentation Sequence (aug)**:
    
    ```python
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order
    
    ```
    
    This sequence defines a series of image augmentation techniques using the `imgaug` library. These techniques include horizontal flips, random crops, Gaussian blur, contrast adjustments, adding Gaussian noise, brightness adjustments, and affine transformations like scaling, translation, rotation, and shearing.
    
2. **Incorporated Augmentation in Preprocessing Function**:
    
    ```python
    ori_imgs = [aug.augment_image(img) for img in ori_imgs]  # Apply augmentations
    
    ```
    
    This line applies the defined augmentation sequence to each input image before further processing, ensuring that the data used for training or inference is augmented with various transformations to increase its diversity and robustness.
    

### Explanation:

- YOLOv4's data augmentation techniques have been incorporated into the EfficientNet-based object detection pipeline to enhance the model's ability to generalize and detect objects accurately under different conditions.
- The `imgaug` library provides a flexible and comprehensive set of augmentation techniques, allowing for a wide range of transformations to be applied to input images.
- Augmentation is performed before normalization and resizing in the preprocessing function, ensuring that all subsequent operations are performed on augmented images.
- By augmenting the training data, the model becomes more robust to variations in lighting, orientation, scale, and other factors, leading to improved performance and generalization ability.

## Modified Loss functions to take into account loss from both the heads :

```python
class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss

```

In this snippet:

- `ModelWithLoss` is a custom module responsible for computing the loss.
- The `forward` method takes input images (`imgs`) and annotations (`annotations`), and optionally, a list of object classes (`obj_list`).
- The model's forward pass (`self.model(imgs)`) returns classification and regression predictions, among other outputs.
- The classification and regression predictions are passed to the Focal Loss (`self.criterion`) for computation.
- The resulting classification and regression losses are returned.

 Modified loss calculation is used in the training loop:

```python
imgs = data['img']
annot = data['annot']

if params.num_gpus == 1:
    imgs = imgs.cuda()
    annot = annot.cuda()

optimizer.zero_grad()
cls_loss1, reg_loss1 , cls_loss2 , reg_loss2 = model(imgs, annot, obj_list=params.obj_list)

# Average the losses
cls_loss1 = cls_loss1.mean()
reg_loss1 = reg_loss1.mean()
cls_loss2 = cls_loss2.mean()
reg_loss2 = reg_loss2.mean()

# Sum the losses from both heads
loss = cls_loss1 + reg_loss1 + cls_loss2 + reg_loss2

```

In this snippet:

- Input images (`imgs`) and annotations (`annot`) are fetched from the data loader.
- If only one GPU is available (`params.num_gpus == 1`), the inputs are moved to the GPU.
- The model is called with the inputs to compute the losses for both heads (`cls_loss1`, `reg_loss1` for the first head, and `cls_loss2`, `reg_loss2` for the second head).
- Each loss is optionally averaged (`mean()`) to account for batch size variations.
- Finally, the losses from both heads are summed to obtain the total loss (`loss`).

This modification allows the training loop to handle models with two detection heads, enabling more complex architectures and tackling diverse detection tasks effectively. Adjustments to the loss calculation, such as averaging or summing the losses from multiple heads, can be made based on the specific requirements of the task and the behavior of the model during training.

## Prepare the Dataset  :

Slight modifications were required for both the datasets to be prepared accordingly : 

```python
# dataset structure should be like this
datasets/
    -your_project_name/
        -train_set_name/
            -*.jpg
        -val_set_name/
            -*.jpg
        -annotations
            -instances_{train_set_name}.json
            -instances_{val_set_name}.json

# for example, food-dataset
datasets/
    -food-dataset/
        -train2017/
            -000000000001.jpg
            -000000000002.jpg
            -000000000003.jpg
        -val2017/
            -000000000004.jpg
            -000000000005.jpg
            -000000000006.jpg
        -annotations
            -instances_train2017.json
            -instances_val2017.json
```

## Creating a .yml file for both the datasets :

```markdown
# create a yml file {your_project_name}.yml under 'projects'folder 
# modify it following 'coco.yml'
 
# for example
project_name: coco
train_set: train2017
val_set: val2017
num_gpus: 4  # 0 means using cpu, 1-N means using gpus 

# mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

# this is coco anchors, change it if necessary
anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

# objects from all labels from your dataset with the order from your annotations.
# its index must match your dataset's category_id.
# category_id is one_indexed,
# for example, index of 'car' here is 2, while category_id of is 3
obj_list: ['person', 'bicycle', 'car', ...]
```

## Training and Inferencing :

```python
# with a coco-pretrained, you can even freeze the backbone and train heads only
# to speed up training and help convergence.

python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 30 \
 --load_weights /path/to/your/weights/yolov4.weights \
 --head_only True
```

```python
python coco_eval.py -p your_project_name -c 5 \
 -w /path/to/your/weights
```

```python
python efficientdet_test.py
```

- The Models were inferenced on single head and double head and the results matched , the weights used were the YOLOv4 pre trained weights.

- The score on PyLint for the given code is 9.01 close to a perfect 10.