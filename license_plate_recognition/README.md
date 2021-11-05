# License Plate Detection

The goal is to localize vehicle's registration plate. 

## Framework

I am using `torch==1.7.1` and `torchvision==0.8.2` to build the model, do the training/evaluation and make predictions.

## Model

The two-stage detector `Faster R-CNN` is used. Specifically, `torchvision` implementation : it's a pre-trained model on the **COCO** object detection dataset, and the model's head is trained from scratch. 

```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def faster_rcnn_pretrained_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
```

## Dataset
## Training and Evaluation configuration
## Metrics
## Image and Video results