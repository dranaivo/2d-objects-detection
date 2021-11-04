# Face Mask Detection

The goal is to determine if a person wears a mask.

## Framework

It's [darknet](https://github.com/AlexeyAB/darknet).

## Model

[YOLO v3]() and [YOLO v4]().

## Dataset

## Training and Evaluation configuration

For full configurations, you can look inside the folder `cfg`.

**Yolo v3**

For `training` :
```ini
batch=64
max_batches = 6000
learning_rate=0.001

width=416
height=416
```

For `evaluation` :
```ini
batch=1
max_batches = 6000
learning_rate=0.001

width=416
height=416
```

**Yolo v4**

For `training` :
```ini
batch=64
max_batches=4000
learning_rate=0.001

width=416
height=416
```

For `evaluation` :
```ini
batch=1
max_batches = 6000
learning_rate=0.001

width=416
height=416
```