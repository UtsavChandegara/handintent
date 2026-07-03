# HandIntent — MVP v1.0

Context-Aware Gesture Intent Recognition System

## What This Is
A software system that detects hand gestures using a standard webcam 
and infers user intent from natural movement — enabling touchless device 
control without any special hardware.

## Architecture
- Layer 1: MediaPipe hand tracking (21 landmarks)
- Layer 2: Rule-based activity context classifier
- Layer 3: BiGRU neural network intent predictor

## Intents Supported
- SELECT — point index finger
- DISMISS — push open palm toward camera
- SCROLL_UP — move hand upward
- SCROLL_DOWN — move hand downward

## Contexts Supported
- READING — both hands open and close together
- NEUTRAL — default state

## How To Run

### Install dependencies
pip3 install mediapipe opencv-python torch numpy boto3 matplotlib

### Collect training data
python3 data_collector.py

### Train BiGRU model
python3 train_bigru.py

### Run live pipeline
python3 main_pipeline.py

### Sync with AWS S3
python3 sync.py

## Tech Stack
Python, MediaPipe, OpenCV, PyTorch, AWS S3, boto3

## Project Status
MVP Complete — v1.0

## Research Context
Final Year Project — Computer Engineering
KSV University, Ahmedabad