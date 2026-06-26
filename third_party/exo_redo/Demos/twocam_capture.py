"""Live camera-based robot state estimation with puck grasping.

This demo continuously:
1. Captures frames from camera
2. Detects ArUco markers on robot and puck
3. Estimates robot joint configuration
4. Runs IK to grasp the puck
5. Displays the result
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mujoco
import cv2
import numpy as np
import mink
import matplotlib.pyplot as plt

camera_device1 = 0  # Change to match your camera
camera_device2 = 1  # Change to match your camera
# Initialize camera
cap1 = cv2.VideoCapture(camera_device1)
if not cap1.isOpened(): raise RuntimeError(f"Failed to open camera device {camera_device1}")
#cap2 = cv2.VideoCapture(camera_device2)
#if not cap2.isOpened(): raise RuntimeError(f"Failed to open camera device {camera_device2}")

while True:
    # Capture frame
    ret1, frame1 = cap1.read()
    #ret2, frame2 = cap2.read()
    #if not ret1 or not ret2: print("Failed to read frame from camera");continue
    if not ret1 : print("Failed to read frame from camera");continue
    rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    #rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    # Detect robot and puck ArUco markers
    # Create overlay
    #display = np.concatenate([rgb1, rgb2], axis=1)
    display = rgb1
    cv2.imshow('Two Camera Capture', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

    waitkey = cv2.waitKey(1) & 0xFF
    if waitkey == ord('q'): break
    elif waitkey == ord('s'): plt.imsave("../redo_mujoco_calibration/random/tmpimgs/kitchen_logi1.png", rgb1);print("saved imgs")