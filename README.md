📄 Project Title:
AI-Powered Hand Gesture Virtual Keyboard using OpenCV and MediaPipe

🧠 Project Description:
This project implements a real-time virtual keyboard controlled by hand gestures. Using a webcam, it detects and tracks hands, particularly the index finger and thumb, to simulate key presses. When the user "pinches" (brings index finger and thumb close together) over a virtual key, it is registered as a key press.

The system is ideal for touchless interaction, useful in medical applications, accessibility tools, kiosks, and public interfaces.

🎯 Key Features:
Real-time hand detection using MediaPipe Hands

Dynamic keyboard interface drawn on webcam feed

Pinch gesture recognition for typing (index + thumb tip)

Supports full QWERTY keyboard layout

Backspace and Space keys included

Typed output is shown on screen

Basic debounce timer to avoid multiple key presses

🛠️ Technical Requirements:
🔹 Languages & Libraries:
Python 3.6+

OpenCV

MediaPipe

NumPy
