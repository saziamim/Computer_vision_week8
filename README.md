# Stereo Vision-Based Object Detection and 3D Position Estimation

This project implements a stereo vision pipeline to detect objects (chairs and tables) and estimate their 3D positions using a pair of images (left and right). It also generates a top-down (bird’s-eye) visualization of detected objects.

---

## Features

- Detects chairs and tables using YOLOv8
- Computes disparity map using StereoSGBM
- Estimates depth from disparity
- Converts image coordinates to 3D positions
- Generates annotated detection images
- Creates a top-down spatial plot

---

## Pipeline Overview

1. Align right image to left image using ORB feature matching  
2. Convert images to grayscale  
3. Compute disparity map using StereoSGBM  
4. Detect objects using YOLOv8  
5. Estimate depth from disparity  
6. Convert to 3D coordinates  
7. Plot object positions in top-down view  

---

## Depth Formula

Depth is computed using:

Z = (fx * baseline) / disparity

Where:
- fx = focal length (pixels)  
- baseline = distance between camera positions (meters)  
- disparity = pixel shift between left and right images  

---

## Installation

Create environment:
conda create -n stereo python=3.10
conda activate stereo

Install dependencies:
conda create -n stereo python=3.10
conda activate stereo


Install dependencies:


pip install opencv-python numpy matplotlib ultralytics


---

## Usage

Run the script:


python stereo_classroom_fixed.py
--left left.png
--right right.png
--output_dir results

---

## Input Requirements

- Two images of the same scene  
- Captured with horizontal camera movement  
- Minimal rotation between shots  
- Good lighting and texture  

---

## Outputs

- aligned_right.png → aligned stereo image  
- disparity.png → disparity visualization  
- detections_and_positions.png → annotated detections  
- topdown_plot.png → bird’s-eye object map  

---


