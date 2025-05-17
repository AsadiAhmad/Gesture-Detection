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

### Step 5: Verify CUDA

for running the code and check the installation we need to verify the cuda with opencv.

```python
print("CUDA Enabled:", cv.cuda.getCudaEnabledDeviceCount())
print("OpenCV Build Info:")
print(cv.getBuildInformation())
```

### Step 6: Download Resources

We need to download the Caffe modele.

We download models from my google drive for protecting the repo in future.

```python
gdown.download(id="1D3ytIZ-ZMMd5MbvVbf2Sn5oZ1L0aQ9IG", output="pose_deploy_linevec_faster_4_stages.prototxt", quiet=False)
gdown.download(id="1f-fCSTg7qFHRVKGIptyPJsgNwRs4XDsK", output="pose_iter_160000.caffemodel", quiet=False)
```

### Step 7: Load Models

We load the Caffe models for pose detection. also we set the cuda here

```python
protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose_iter_160000.caffemodel"

net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
```

### Step 8: Set Body Points

in here we define the structure of the body skeleton by connecting the points of the body

```python
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
```

### Step 9: Convert image to blob

This section works with GPU

```python
def convert_image_to_blob(frame):
    return cv.dnn.blobFromImage(frame, 1.0/255, (256, 256), (0, 0, 0), swapRB=False, crop=False)
```

### Step 10: Run Inference Async (forward pass)

In this section we run move forward through the model for pose detection. this section used GPU.

```python
def run_inference():
    if not hasattr(run_inference, '_warmed_up'):
        net.forward()
        run_inference._warmed_up = True

    return net.forward()
```

### Step 11: Extract points

In this section we extract all body points (15 points here). this section use CPU

```python
def extract_keypoints(output, height, width):
    points = []
    for i in range(15):
        probMap = output[0, i, :, :]
        _, _, _, max_loc = cv.minMaxLoc(probMap)
        points.append((int(max_loc[0] * width / output.shape[3]), 
                       int(max_loc[1] * height / output.shape[2])))
    return points
```

### Step 12: Display Points & Skeleton

in this section we use the POSE_PIARS to draw the skeleton of the body. this section uses CPU.

```python
# CPU based
def draw_skeleton(frame, points, POSE_PAIRS): 
    image_skeleton = frame.copy()

    for pair in POSE_PAIRS:
        partA, partB = pair
        if points[partA] and points[partB]:
            cv.line(image_skeleton, points[partA], points[partB], (255, 255, 0), 2)
            cv.circle(image_skeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv.FILLED)
    return image_skeleton
```

### Step 13: Classifying Gesture

This section tries to classigying the gesture with points of the body. this section uses CPU.

```python
def calculate_angle(line):
    point1 = line[0]
    point2 = line[1]
    if not point1 or not point2:
        return None
    
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
        
    return angle_deg
```

```python
def classify_pose(points): # CPU based
    try:
        # each points od body are (width, height)
        head = points[0]
        neck = points[1]
        right_shoulder = points[2]
        right_elbow = points[3]
        right_wrist = points[4]
        left_shoulder = points[5]
        left_elbow = points[6]
        left_wrist = points[7]
        right_hip = points[8]
        right_knee = points[9]
        right_ankle = points[10]
        left_hip = points[11]
        left_knee = points[12]
        left_ankle = points[13]
        center = points[14]

        # each line of the body
        head_line = (head, neck)
        right_shoulder_line = (neck, right_shoulder)
        left_shoulder_line = (neck, left_shoulder)
        torso_line = (neck, center)

        right_upper_arm_line = (right_shoulder, right_elbow)
        right_lower_arm_line = (right_elbow, right_wrist)
        left_upper_arm_line = (left_shoulder, left_elbow)
        left_lower_arm_line = (left_elbow, left_wrist)

        right_thigh_line = (right_hip, right_knee)
        right_shin_line = (right_knee, right_ankle)
        left_thigh_line = (left_hip, left_knee)
        left_shin_line = (left_knee, left_ankle)

        right_hip_line = (center, right_hip)
        left_hip_line = (center, left_hip)

        if not neck or not center:
            return "none"
        
        # detecting Position
        torso_angle = calculate_angle(torso_line)
        vertical_torso = False
        horizental_torso = False
        if (70 < torso_angle < 110):
            vertical_torso = True
        if (150 < torso_angle < 210) or (330 < torso_angle) or (torso_angle < 30):
            horizental_torso = True

        right_thigh_angle = calculate_angle(right_thigh_line)
        vertical_right_thigh = False
        horizental_right_thigh = False
        if (70 < right_thigh_angle < 110):
            vertical_right_thigh = True
        if (150 < right_thigh_angle < 210) or (330 < right_thigh_angle) or (right_thigh_angle < 30):
            horizental_right_thigh = True

        left_thigh_angle = calculate_angle(left_thigh_line)
        vertical_left_thigh = False
        horizental_left_thigh = False
        if (70 < left_thigh_angle < 110):
            vertical_left_thigh = True
        if (150 < left_thigh_angle < 210) or (330 < left_thigh_angle) or (left_thigh_angle < 30):
            horizental_left_thigh = True
        
        # Standing gesture
        if vertical_torso and vertical_right_thigh and vertical_left_thigh:
            return "standing"

        # Sitting gesture
        if horizental_left_thigh and horizental_right_thigh and vertical_torso:
            return "sitting"

        # Laying gesture
        if horizental_torso:
            return "laying"

    except:
        return "none"

    return "none"
```

### Step 14: Put every thing together

```python
def detect_gesture(frame, POSE_PAIRS):
    blob = convert_image_to_blob(frame)
    net.setInput(blob)
    output = run_inference()
    points = extract_keypoints(output, frame.shape[0], frame.shape[1])
    frame_with_skeleton = draw_skeleton(frame, points, POSE_PAIRS)
    label = classify_pose(points)
    return frame_with_skeleton, label
```

### Step 15: Run Live Gesture Detection

If you want to use laptop camera use the `cap = cv.VideoCapture(0)` code if you want to use your phone camera you can install IP Webcam and use the second code `cap = cv.VideoCapture('http://21.118.71.170:8080/video', cv.CAP_FFMPEG)`.

```python
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('http://21.118.71.170:8080/video', cv.CAP_FFMPEG)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
skip_frames = 3

prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (640, 480))
    output_frame, label = detect_gesture(frame, POSE_PAIRS)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv.putText(output_frame, f"FPS: {fps:.1f}", (520, 30), 
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv.putText(output_frame, label, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.imshow('Gesture Detection', output_frame)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
cap.release()
```

## 🪪 License

This project is licensed under the MIT License.
