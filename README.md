# Hand Gesture Recognition (HGR)

A real-time Hand Gesture Recognition system built using **Python, OpenCV, and MediaPipe**.  
The program detects hand landmarks through the webcam and classifies gestures such as:

- ✊ Fist  
- 🖐️ Open Palm  
- ✌️ Victory  
- 👍 Thumbs Up / Thumbs Down  
- 👆 One (Index Finger)  
- 🤘 Rock  
- 👌 OK  
- 0–5 finger counts  

This project uses **21 MediaPipe hand landmarks** and rule-based logic to detect gestures accurately.


## 🚀 Features
- Real-time webcam detection  
- Smooth and stable gesture prediction using history buffers  
- Multiple gesture classifications  
- Lightweight (no training required)


## 📦 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the program:
python hgr.py
