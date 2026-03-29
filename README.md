FaceScan — Real-Time Face Recognition & Audio Memory System

A real-time intelligent system that recognizes faces, speaks identities, and remembers interactions over time — combining computer vision, speech processing, and cloud AI into a seamless human-like experience.

Overview

FaceScan is not just a face recognition tool — it behaves like a memory system.

It can:

Recognize people in real time
Speak their name naturally
Recall when & where it last saw them
Learn new faces through voice input
Persist memory across sessions
Key Features
Real-Time Face Detection via webcam
Hybrid Recognition System
Fast local detection (LBPH)
Cloud verification (AWS Rekognition)
Voice-Based Learning
Records user’s name via microphone
Cleans speech using AWS Comprehend
Human-like Voice Output
ElevenLabs TTS integration
Persistent Memory
SQLite database stores identities, timestamps, and locations
Smart Cooldown System
Avoids repetitive announcements (2-minute buffer)
Dynamic Image Updates
Continuously improves recognition accuracy
Cross-Platform Support
macOS, Windows, Raspberry Pi ready
Tech Stack
Layer	Technology
Language	Python 3.13
Face Detection	OpenCV (Haar Cascade)
Local Recognition	OpenCV (LBPH)
Cloud Recognition	AWS Rekognition
NLP (Name Extraction)	AWS Comprehend
Speech-to-Text	Google STT (SpeechRecognition)
Text-to-Speech	ElevenLabs API
Audio Handling	pygame, sounddevice, scipy
Database	SQLite
Location Tracking	geocoder (IP-based)
🧠 System Architecture
Camera Input
   ↓
Face Detection
   ↓
Local Recognition (LBPH)
   ├── High Confidence → Speak Name
   └── Low Confidence → AWS Rekognition
           ├── Match Found → Speak Name
           └── New Face
                 ↓
           Record Voice (Name)
                 ↓
           Clean Text (AWS Comprehend)
                 ↓
           Store in Database
⚙️ Setup Guide
1️Clone the Repository
git clone https://github.com/your-username/face-memory-system.git
cd face-memory-system
2️Install Dependencies
pip install opencv-python numpy sounddevice scipy requests geocoder pygame boto3 python-dotenv SpeechRecognition
 Configure Environment Variables

Create a .env file in the root directory:

ELEVENLABS_API_KEY=your_elevenlabs_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=us-east-1
 Run the Application
python face_audio_system.py

Press Q to quit.

How It Works (Execution Flow)
Webcam captures frame
Face is detected
System checks:
Known face? → Speak name + last seen info
Unknown face? → Record voice → extract name → store
Memory is updated continuously
Cooldown prevents repeated announcements
Project Structure
face-memory-system/
├── face_audio_system.py   # Main application
├── face_memory.db         # SQLite database (auto-generated)
├── lbph_model.yml         # Trained face model
├── seen_log.json          # Cooldown + timestamps
├── face_images/           # Stored face images
├── audio_clips/           # Recorded audio samples
└── .env                   # API keys (not committed)
Reset the Database
macOS / Linux
rm face_memory.db seen_log.json lbph_model.yml
rm -rf face_images/* audio_clips/*
Windows (PowerShell)
Remove-Item -Force face_memory.db, seen_log.json, lbph_model.yml
Remove-Item -Recurse -Force face_images\*, audio_clips\*
Update a Person’s Name
python3 -c "
import sqlite3
con = sqlite3.connect('face_memory.db')
con.execute(\"UPDATE persons SET transcript=? WHERE id=?\", ('Your Name', 1))
con.commit()
con.close()
"
Advanced Features
Parallel AWS Calls for faster recognition
Continuous Learning System
Memory Persistence Across Sessions
Location-Aware Recognition (IP-based)
Accuracy Boost via Dynamic Image Updates
Use Cases
Smart assistants
Security & surveillance
Networking events (auto-name recall)
Accessibility tools
Personalized AI companions
Future Improvements
Mobile app integration
Edge deployment optimization (Jetson / Pi)
Multi-person conversation tracking
Emotion detection
Face embedding upgrade (DeepFace / FaceNet)
Contributing

Pull requests are welcome! For major changes, please open an issue first.

License

MIT License

Final Note

This project showcases the intersection of:

Computer Vision
NLP
Speech Processing
Real-Time Systems
Cloud AI
