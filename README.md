# AI Posture Monitor

## GOAL

Using a fixed webcam, one can detect the quality of their posture and their presence (or lack thereof). I aim to classify the user's sitting state in real time, with physical feedback via an Arduino-controlled RGB LED.

## How It Will Work

The webcam will track faces and poses. It will start with a short calibration (to establish a baseline for a good sitting posture), after which the state will be classified into either:

- **Good Posture:** LED off
- **Slouching:** LED red
- **Away:** LED blue

## Architecture

## Setup (CURRENTLY UNAVAILABLE)

To run:
```bash
pip install -r requirements.txt
python python/main.py
```

## Status