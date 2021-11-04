# Face Mask Detection

The goal is to determine if a person wears a mask.

## Framework

It's [darknet](https://github.com/AlexeyAB/darknet).

## Model

[YOLO v3]() and [YOLO v4]().

## Dataset

## Training and Evaluation configuration

**Yolo v3**

For `training` :
```ini
# training
batch=64
subdivisions=16
max_batches = 6000

# optimizer
learning_rate=0.001
momentum=0.9
decay=0.0005
burn_in=100
policy=steps
steps=4800,5400
scales=.1,.1

# pre-processing and augmentation
width=416
height=416
channels=3
angle=0
saturation=1.5
exposure=1.5
hue=.1
```

For `evaluation` :
```ini
# testing
batch=1
subdivisions=1
max_batches = 6000

# optimizer
learning_rate=0.001
momentum=0.9
decay=0.0005
burn_in=100
policy=steps
steps=4800,5400
scales=.1,.1

# pre-processing and augmentation
width=416
height=416
channels=3
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
```

**Yolo v4**

For `training` :
```ini
# training
batch=64
subdivisions=16
max_batches=4000

# optimizer
learning_rate=0.001
momentum=0.949
decay=0.0005
burn_in=150
policy=steps
steps=2800,3600
scales=.1,.1

# pre-processing and data augmentation
width=416
height=416
channels=3
angle=0
saturation=1.5
exposure=1.5
hue=.1
#cutmix=1
mosaic=1
```

For `evaluation` :
```ini
# testing
batch=1
subdivisions=1
max_batches = 6000

# optimizer
learning_rate=0.001
momentum=0.949
decay=0.0005
burn_in=100
policy=steps
steps=800
scales=.1,.1

# pre-processing and data augmentation
width=416
height=416
channels=3
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
#cutmix=1
mosaic=1
```