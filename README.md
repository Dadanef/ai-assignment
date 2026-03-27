# Facial Emotion Recognition Project

## Overview
This project implements:
- Partner A: preprocessing, face detection tuning, real-time webcam mode
- Partner B: emotion classification, rich visualization, batch accuracy analysis

## Requirements
- Python 3.10+
- Webcam for webcam mode

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
```
emotion_project/
├── main.py
├── partner_a.py
├── classifier.py
├── visualize.py
├── batch_analysis.py
├── config.py
├── requirements.txt
├── images/
├── labeled_dataset/
└── output/
```

## Dataset format

```
labeled_dataset/
├── angry/
├── disgust/
├── fear/
├── happy/
├── sad/
├── surprise/
└── neutral/
```
# BASH
PARTNER A
``````
python main.py --mode a1 --image images/test.jpg
python main.py --mode a2 --image images/test.jpg
python main.py --mode webcam --skip 3
``````
PARTNER B
``````
python main.py --mode classify --image images/test.jpg
python main.py --mode visualize --image images/test.jpg
python main.py --mode batch --dataset labeled_dataset
``````