# Gesture-Detection
Real-time Gesture Detection using CUDA-accelerated OpenCV in Python. Leverages GPU for high-performance image processing tasks, ensuring efficient and responsive gesture recognition.

<div display=flex align=center>
  <img src="/Gif/Gesture.gif" width="600px"/>
</div>

## Tech :hammer_and_wrench: Languages and Tools :

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/jupyter/jupyter-original.svg" title="Jupyter Notebook" alt="Jupyter Notebook" width="40" height="40"/>&nbsp;
  <img src="https://assets.st-note.com/img/1670632589167-x9aAV8lmnH.png" title="Google Colab" alt="Google Colab" width="40" height="40"/>&nbsp;
  <img src="https://github.com/AsadiAhmad/AsadiAhmad/blob/main/Logo/Python/math.png" title="Math" alt="Math" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" title="OpenCV" alt="OpenCV" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="Numpy" alt="Numpy" width="40" height="40"/>&nbsp;
  <img src="https://www.svgrepo.com/show/373541/cuda.svg" title="CUDA" alt="CUDA" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/matplotlib/matplotlib-original.svg"  title="MatPlotLib" alt="MatPlotLib" width="40" height="40"/>&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/1200px-Google_Drive_icon_%282020%29.svg.png"  title="Gdown" alt="Gdown" width="40" height="40"/>&nbsp;
</div>

- Python : Popular language for implementing Neural Network
- Jupyter Notebook : Best tool for running python cell by cell
- Google Colab : Best Space for running Jupyter Notebook with hosted server
- Math : Essential Python library for basic mathematical operations and functions
- OpenCV : Best Library for working with images
- Numpy : Best Library for working with arrays in python
- CUDA : used for NVIDIA GPU acceleration and get better frame rate
- MatPlotLib : Library for showing the charts in python
- GDown : Download Resources from Google Drive

## 💻 Run the Notebook on Google Colab

You can easily run this code on google colab by just clicking this badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/AsadiAhmad/Gesture-Detection/blob/main/Code/Gesture_Detection.ipynb)

Live version not have google colab version because the google colab will crash.

## Models

We have used caffe Open Pose model. we have the code for donwloading the model.

```python
gdown.download(id="1D3ytIZ-ZMMd5MbvVbf2Sn5oZ1L0aQ9IG", output="pose_deploy_linevec_faster_4_stages.prototxt", quiet=False)
gdown.download(id="1f-fCSTg7qFHRVKGIptyPJsgNwRs4XDsK", output="pose_iter_160000.caffemodel", quiet=False)
```

## 📝 Tutorial

### Step 1: Install CUDA toolkit
The first steps are haveing a NVIDIA GPU and download and install the CUDA Toolki (I have used the version 11.7.0) [![CUDA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)

Check CUDA version:
```sh
nvcc --version
```

### Step 2: Install cuDNN
After that you should donwload and install the cuDN (I have used the version 8.6.0 for 11.X CUDA) [![cuDNN](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)](https://developer.nvidia.com/rdp/cudnn-archive)

After these installation make sure you move the dll files from the cuDNN zip file into the CUDA installation path.

### Step 3: Install OpenCV with CMake
Now you have installed the CUDA and cuDNN now we use the Visual studio and CMake to build our OpenCV with CUDA. you can watch this video [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/watch?v=5NwU1MmmqWo)

### Step 4: Import Libraries

we need to import these libraries :

`math`, `numpy`, `cv2`, `gdown`, `time`

```python
import math
import numpy as np
import cv2 as cv
import gdown
import time
```



## 🪪 License

This project is licensed under the MIT License.
