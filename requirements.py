# Face Memory System – Python dependencies
# Install with: pip install -r requirements.txt

# opencv-python
# face_recognition
# numpy
# sounddevice
# scipy
# requests
# geocoder
# pygame
# elevenlabs          # optional SDK, not used directly but useful for reference

# Raspberry Pi only (install separately via apt + pip):
# sudo apt install python3-picamera2 libcamera-apps
# pip install picamera2





# Remove-Item -Force "face_memory.db" -ErrorAction SilentlyContinue
# Remove-Item -Force "lbph_model.yml" -ErrorAction SilentlyContinue
# Remove-Item -Recurse -Force "face_images" -ErrorAction SilentlyContinue
# Remove-Item -Recurse -Force "audio_clips" -ErrorAction SilentlyContinue