# camera-helmet-detection
Final work developed for the Topics in Image Processing subject at UFMS.
This version uses the model RTMDet tiny.

## Dependencies
- python 3.11.6
- mmdet 3.2.0
- cv2 4.7.0

## How to use

To use the model with your GPU (uses "cuda:0" as default):
```kernel
python3 run.py cuda
```

To use the model with your CPU:
```kernel
python3 run.py cpu
```
