# RatinaChess

A Python-based eye-gaze and hand-tracking chess interface that enables you to interact with the chess game through camera-based gaze and hand movements.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Eye-Gaze Tracking**: Uses computer vision to detect where the user is looking on the board.
- **Hand Tracking**: Enables intuitive piece selection and movement using hand gestures.
- **Chess Game Logic**: Interactive game module for gameplay management.
- **Calibration Support**: Ensures precise tracking accuracy with built-in calibration routines.
- **Modular Design**: Reusable modules for gaze, hand, and facial mesh tracking.

---

## Installation

1. Clone the repository:
   git clone https://github.com/Sarthak1311/ratinaChess.git
   cd ratinaChess
2. Install dependencies:
  pip install -r requirements.txt


## Usage
1. Calibration
Run the eye-gaze calibration script to align tracking to your camera setup:
      python calibration_eye_gaze.py
2. Start Gaze or Hand Tracking
    To track eye movements:
      python GazePrediction.py
3. To track hand gestures:
    python handTracking.py
4. Play Chess
    python game.py

---

##vFile Structure

- **GazePrediction.py	Module for predicting gaze direction.
- **calibration_eye_gaze.py & .csv	Scripts and data for eye-gaze calibration.
- **eyetracking.py, gazetracking.py	Support modules for different tracking methods.
- **handTracking.py	Hand gesture detection for interactions.
- **facemesh.py	Facial landmark detection module.
- **game.py	Main chess game logic and loop.
- **template.py	Template or helper utilities.
- **requirements.txt	Required Python packages.
- **images/, calibration_data/	Assets and calibration samples.

---

## Requirements
- Ensure you have the following in your environment:
1. Python 3.7 or later
2. OpenCV (opencv-python)
3. MediaPipe
4. Numpy
5. Pandas
6. Other dependencies listed in requirements.txt


---  
## Contributing
Contributions are welcome! If you'd like to add features or fix bugs, please follow these steps:
1. Fork the repository
2. Create a feature branch:
git checkout -b feature/my-new-feature
3. Commit your changes:
git commit -m "Add new feature"
4. Push to your branch:
git push origin feature/my-new-feature
Open a Pull Request
