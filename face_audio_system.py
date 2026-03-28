"""
Face Recognition + Audio Memory System
=======================================
- OpenCV Haar Cascade for face detection
- LBPH for fast local recognition
- AWS Rekognition as backup for accurate person clustering
- pyttsx3 for TTS (free, offline)
- Google STT for speech recognition

Install:
    pip install opencv-python numpy sounddevice scipy requests geocoder pygame boto3 speechrecognition python-dotenv
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import sqlite3
import datetime
import threading

import cv2
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import requests
import geocoder
import pygame
import boto3
import speech_recognition as sr

# ── AWS Rekognition ───────────────────────────────────────────────────────────
rekognition = boto3.client(
    "rekognition",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
AWS_SIMILARITY_THRESHOLD = 85

# ── Tuning knobs ──────────────────────────────────────────────────────────────
RECORD_SECONDS    = 3
SAMPLE_RATE       = 44_100
REDETECT_COOLDOWN = 40  # 2 minutes
DB_PATH           = "face_memory.db"
AUDIO_DIR         = "audio_clips"
FACE_DIR          = "face_images"
LBPH_MODEL_PATH   = "lbph_model.yml"
LBPH_THRESHOLD    = 65

# ── OpenCV face detector ──────────────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ── LBPH recogniser ───────────────────────────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer_trained = False

# ── Runtime state ─────────────────────────────────────────────────────────────
recently_seen: dict = {}
SEEN_LOG_PATH = "seen_log.json"

# Tracks which cooldown_keys currently have an active thread running.
# This replaces the old global `processing` boolean which blocked ALL faces.
active_keys: set = set()
active_keys_lock = threading.Lock()

# LBPH lock — prevents thread-unsafe concurrent OpenCV calls
lbph_lock = threading.Lock()  # ← ADD THIS HERE

# Speech lock — only one voice at a time, no overlapping ever
speech_lock = threading.Lock()


def load_seen_log():
    global recently_seen
    if os.path.exists(SEEN_LOG_PATH):
        import json
        with open(SEEN_LOG_PATH) as f:
            recently_seen = json.load(f)
            recently_seen = {int(k): v for k, v in recently_seen.items()}

def save_seen_log():
    import json
    with open(SEEN_LOG_PATH, "w") as f:
        json.dump(recently_seen, f)

# ─────────────────────────────────────────────────────────────────────────────
# AWS Rekognition — smart clustering
# ─────────────────────────────────────────────────────────────────────────────

def aws_same_person(img_path_1: str, img_path_2: str) -> tuple:
    """Compare two face images via AWS. Returns (is_match, confidence)."""
    try:
        for path in [img_path_1, img_path_2]:
            img = cv2.imread(path)
            if img is None:
                print(f"  AWS: cannot read {path}")
                return False, 0.0
            h, w = img.shape[:2]
            print(f"  AWS: image size {w}x{h} for {path}")
            if h < 300 or w < 300:
                scale = max(300 / h, 300 / w)
                img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(path, img)
                print(f"  AWS: upscaled to {int(w*scale)}x{int(h*scale)}")

        with open(img_path_1, "rb") as f1:
            src = f1.read()
        with open(img_path_2, "rb") as f2:
            tgt = f2.read()

        response = rekognition.compare_faces(
            SourceImage={"Bytes": src},
            TargetImage={"Bytes": tgt},
            SimilarityThreshold=AWS_SIMILARITY_THRESHOLD,
            QualityFilter="LOW",
        )

        if response["FaceMatches"]:
            return True, response["FaceMatches"][0]["Similarity"]

        # Log unmatched faces to help debug
        print(f"  AWS: no match. FaceMatches={len(response['FaceMatches'])}, "
              f"UnmatchedFaces={len(response.get('UnmatchedFaces', []))}")
        return False, 0.0

    except rekognition.exceptions.InvalidParameterException as e:
        print(f"  AWS: no face detected in image — {e}")
        return False, 0.0
    except Exception as e:
        print(f"  AWS error: {e}")
        return False, 0.0


def find_person_cluster(new_img_path: str) -> int | None:
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, rep_image_path FROM persons ORDER BY id ASC"
    ).fetchall()
    con.close()

    print(f"  Checking against {len(rows)} existing person(s) in DB ...")
    for person_id, rep_path in rows:
        if not rep_path:
            print(f"  Person #{person_id} has no rep_image_path, skipping.")
            continue
        if not os.path.exists(rep_path):
            print(f"  Person #{person_id} rep image missing on disk: {rep_path}")
            continue
        is_match, confidence = aws_same_person(new_img_path, rep_path)
        print(f"  AWS: person #{person_id} → {confidence:.1f}% {'✓ MATCH' if is_match else '✗'}")
        if is_match:
            return person_id

    print("  AWS: no match found — treating as new person.")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Name extraction — phrase stripping
# ─────────────────────────────────────────────────────────────────────────────

def extract_name_from_transcript(transcript: str) -> str:
    """Strip common intro phrases and return just the name."""
    if not transcript:
        return ""

    cleaned = transcript.lower()

    # Pattern 1: name comes AFTER the phrase — "my name is John"
    phrases_before_name = [
        "hi my name is", "hello my name is", "hey my name is",
        "my name is", "the name is", "name is",
        "i am", "i'm", "they call me", "call me",
        "it's", "its", "hi i'm", "hello i'm",
    ]
    for phrase in phrases_before_name:
        if phrase in cleaned:
            idx = cleaned.index(phrase) + len(phrase)
            remainder = transcript[idx:].strip()
            name = remainder.strip(".,!?").capitalize() if remainder else ""
            if name:
                print(f"  Extracted name: {name!r}")
                return name

    # Pattern 2: name comes BEFORE the phrase — "John is my name"
    phrases_after_name = [
        " is my name", " is the name",
    ]
    for phrase in phrases_after_name:
        if phrase in cleaned:
            idx = cleaned.index(phrase)
            name = transcript[:idx].strip().strip(".,!?").capitalize()
            if name:
                print(f"  Extracted name: {name!r}")
                return name

    print("  Could not extract name, saving full transcript.")
    return transcript

# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(FACE_DIR,  exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            first_seen     TEXT    NOT NULL,
            location       TEXT    NOT NULL,
            rep_image_path TEXT,
            audio_path     TEXT,
            transcript     TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS face_images (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id  INTEGER NOT NULL,
            image_path TEXT    NOT NULL,
            captured   TEXT    NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(id)
        )
    """)
    con.commit()
    con.close()


def load_all_persons():
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, transcript, first_seen, location FROM persons"
    ).fetchall()
    con.close()
    return rows


def create_person(location: str, rep_image_path: str) -> int:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(DB_PATH)
    cur = con.execute(
        "INSERT INTO persons (first_seen, location, rep_image_path) VALUES (?, ?, ?)",
        (now, location, rep_image_path),
    )
    pid = cur.lastrowid
    con.commit()
    con.close()
    return pid


def add_face_image(person_id: int, image_path: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO face_images (person_id, image_path, captured) VALUES (?, ?, ?)",
        (person_id, image_path, now),
    )
    con.commit()
    con.close()

# ─────────────────────────────────────────────────────────────────────────────
# LBPH
# ─────────────────────────────────────────────────────────────────────────────

def retrain_recognizer():
    global recognizer_trained
    labels, images = [], []
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT person_id, image_path FROM face_images").fetchall()
    con.close()
    for person_id, img_path in rows:
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(cv2.resize(img, (200, 200)))
        labels.append(person_id)
    with lbph_lock:
        if images:
            recognizer.train(images, np.array(labels))
            recognizer.save(LBPH_MODEL_PATH)
            recognizer_trained = True
            print(f"  LBPH retrained on {len(images)} image(s), {len(set(labels))} person(s).")
        else:
            recognizer_trained = False


def load_recognizer():
    global recognizer_trained
    if os.path.exists(LBPH_MODEL_PATH):
        recognizer.read(LBPH_MODEL_PATH)
        recognizer_trained = True
        print("  LBPH model loaded.")


def lbph_predict(gray_crop) -> tuple:
    with lbph_lock:
        if not recognizer_trained:
            return None, 999
        try:
            label, conf = recognizer.predict(cv2.resize(gray_crop, (200, 200)))
            return label, conf
        except Exception:
            return None, 999

# ─────────────────────────────────────────────────────────────────────────────
# Location
# ─────────────────────────────────────────────────────────────────────────────

def get_location() -> str:
    try:
        g = geocoder.ip("me")
        if g.ok:
            parts = [p for p in [g.city, g.state, g.country] if p]
            return ", ".join(parts) if parts else "Unknown location"
    except Exception:
        pass
    return "Unknown location"

# ─────────────────────────────────────────────────────────────────────────────
# Audio
# ─────────────────────────────────────────────────────────────────────────────

def beep(frequency: int = 880, duration_ms: int = 350):
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1)
    n    = int(SAMPLE_RATE * duration_ms / 1000)
    t    = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    mono = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    pygame.sndarray.make_sound(np.column_stack([mono, mono])).play()
    pygame.time.wait(duration_ms + 60)


def record_audio(seconds: int = RECORD_SECONDS) -> np.ndarray:
    print(f"  Recording for {seconds}s ... speak now!")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16")
    sd.wait()
    print("  Recording done.")
    return audio


def save_audio(audio: np.ndarray, person_id: int) -> str:
    path = os.path.join(AUDIO_DIR, f"person_{person_id}_{int(time.time())}.wav")
    wav_write(path, SAMPLE_RATE, audio)
    return path


def speak(text: str):
    print(f"  Speaking: {text}")
    try:
        import subprocess
        subprocess.run(["powershell", "-Command",
            f'Add-Type -AssemblyName System.Speech; '
            f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
            f'$s.Speak("{text}")'], check=True)
    except Exception as e:
        print(f"  TTS error: {e}")


def transcribe_audio(audio_path: str) -> str:
    recognizer_sr = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer_sr.record(source)
        transcript = recognizer_sr.recognize_google(audio_data)
        print(f"  Google STT: {transcript!r}")
        return transcript
    except sr.UnknownValueError:
        print("  STT: could not understand audio")
        return ""
    except Exception as e:
        print(f"  STT error: {e}")
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

def build_camera():
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}))
        picam2.start()
        time.sleep(1)
        def grab():
            return cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
        print("Using Picamera2 (Raspberry Pi)")
        return grab, picam2.stop
    except Exception:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("No camera found.")
        def grab():
            ok, frame = cap.read()
            return frame if ok else None
        print("Using OpenCV webcam (PC)")
        return grab, cap.release

# ─────────────────────────────────────────────────────────────────────────────
# Main face handler
# ─────────────────────────────────────────────────────────────────────────────

# ── Speech lock — only one voice at a time, no overlapping ───────────────────
speech_lock = threading.Lock()

def handle_face(gray_crop, color_crop, location, known_persons_ref, cooldown_key, full_frame):
    """
    1. LBPH quick local check
    2. If uncertain → AWS Rekognition using FULL FRAME (not crop)
    3. If no match anywhere → new person
    Always releases cooldown_key from active_keys when done.
    Speech is serialized via speech_lock — no overlapping audio ever.
    """
    global recognizer_trained

    # Unique ID per thread — prevents timestamp collisions between threads
    unique_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(threading.current_thread())}"
    temp_crop_path  = os.path.join(FACE_DIR, f"temp_crop_{unique_id}.jpg")
    temp_frame_path = os.path.join(FACE_DIR, f"temp_frame_{unique_id}.jpg")

    cv2.imwrite(temp_crop_path,  color_crop)  # for LBPH training
    cv2.imwrite(temp_frame_path, full_frame)  # for AWS — full photo with face context

    try:
        # Step 1: LBPH
        lbph_id, lbph_conf = lbph_predict(gray_crop)
        print(f"\n  LBPH: person #{lbph_id}, conf {lbph_conf:.1f}")

        person_id = None

        if lbph_id is not None and lbph_conf < LBPH_THRESHOLD:
            # Validate the ID actually exists in the current DB
            # guards against stale LBPH model returning IDs from cleared DB
            con = sqlite3.connect(DB_PATH)
            exists = con.execute("SELECT 1 FROM persons WHERE id=?", (lbph_id,)).fetchone()
            con.close()
            if exists:
                person_id = lbph_id
                print(f"  LBPH confident → person #{person_id}")
            else:
                print(f"  LBPH returned invalid ID #{lbph_id} (not in DB) → falling back to AWS")
                person_id = find_person_cluster(temp_frame_path)
        else:
            print("  LBPH uncertain → AWS Rekognition ...")
            person_id = find_person_cluster(temp_frame_path)  # full frame for AWS

        if person_id is not None:
            # Known person
            final_crop_path = os.path.join(FACE_DIR, f"person_{person_id}_{unique_id}.jpg")
            os.rename(temp_crop_path, final_crop_path)
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
            add_face_image(person_id, final_crop_path)
            retrain_recognizer()

            match = next((r for r in known_persons_ref if r[0] == person_id), None)
            if match:
                # Double cooldown check — guards against LBPH threshold
                # fluctuations causing cooldown_key to flip between slots
                last_spoken = recently_seen.get(person_id, 0)
                if time.time() - last_spoken < REDETECT_COOLDOWN:
                    print(f"  Cooldown active for person #{person_id}, skipping speech.")
                else:
                    recently_seen[person_id] = time.time()
                    _, transcript, first_seen, loc_stored = match
                    dt            = datetime.datetime.strptime(first_seen, "%Y-%m-%d %H:%M:%S")
                    friendly_date = dt.strftime("%B %d, %Y at %I:%M %p")
                    name   = transcript if transcript else "stranger"
                    speech = f"{name}, met on {friendly_date}, location {loc_stored}."
                    print(f"\nKnown person #{person_id}!")
                    with speech_lock:
                        speak(speech)
                    save_seen_log()
        else:
            # New person
            print("\nNew person!")

            pid = create_person(location, temp_frame_path)

            final_crop_path  = os.path.join(FACE_DIR, f"person_{pid}_{unique_id}.jpg")
            final_frame_path = os.path.join(FACE_DIR, f"person_{pid}_{unique_id}_rep.jpg")
            os.rename(temp_crop_path,  final_crop_path)
            os.rename(temp_frame_path, final_frame_path)

            # Store full frame as rep so AWS can reliably find the face
            con = sqlite3.connect(DB_PATH)
            con.execute("UPDATE persons SET rep_image_path=? WHERE id=?", (final_frame_path, pid))
            con.commit()
            con.close()

            add_face_image(pid, final_crop_path)
            retrain_recognizer()

            with speech_lock:
                speak("New face detected. Please say your name after the beep.")
                time.sleep(0.2)
                beep()
                audio = record_audio(RECORD_SECONDS)

            audio_path = save_audio(audio, pid)

            print("  Transcribing audio ...")
            raw_transcript = transcribe_audio(audio_path)
            print(f"  Raw transcript: {raw_transcript!r}")
            transcript = extract_name_from_transcript(raw_transcript)
            print(f"  Saved name: {transcript!r}")

            con = sqlite3.connect(DB_PATH)
            con.execute("UPDATE persons SET audio_path=?, transcript=? WHERE id=?",
                        (audio_path, transcript, pid))
            con.commit()
            con.close()

            recently_seen[pid] = time.time()
            save_seen_log()
            known_persons_ref[:] = load_all_persons()

    except Exception as e:
        print(f"  handle_face error: {e}")
        for p in [temp_crop_path, temp_frame_path]:
            if os.path.exists(p):
                os.remove(p)

    finally:
        # Always release — even if exception occurred
        with active_keys_lock:
            active_keys.discard(cooldown_key)

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────


def main():
    init_db()
    load_seen_log()
    pygame.init()
    load_recognizer()

    grab_frame, release_camera = build_camera()
    location      = get_location()
    known_persons = load_all_persons()

    print(f"Location : {location}")
    print(f"Known persons in DB: {len(known_persons)}")
    print("Running ... press Q to quit.\n")

    # Per-face cooldown tracking.
    # Known faces  : keyed by person ID (integer)
    # Unknown faces: keyed by screen region "unknown_X_Y" so multiple
    #                strangers in different parts of the frame each get
    #                their own independent cooldown slot.
    last_triggered: dict = {}

    while True:
        frame = grab_frame()
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now  = time.time()

        raw_faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        # Deduplicate overlapping boxes so the same face isn't detected twice
        if len(raw_faces) > 1:
            raw_as_xywh = [[x, y, w, h] for (x, y, w, h) in raw_faces]
            raw_doubled  = raw_as_xywh * 2  # groupRectangles needs each rect twice
            merged, _    = cv2.groupRectangles(raw_doubled, 1, 0.3)
            faces_rect   = merged if len(merged) > 0 else raw_faces
        else:
            faces_rect = raw_faces

        for (x, y, fw, fh) in faces_rect if len(faces_rect) > 0 else []:
            x2, y2 = x + fw, y + fh
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            gray_crop  = gray[y:y2, x:x2]
            color_crop = frame[y:y2, x:x2]
            if gray_crop.size == 0:
                continue

            lbph_id, lbph_conf = lbph_predict(gray_crop)

            if lbph_id is not None and lbph_conf < LBPH_THRESHOLD:
                label        = f"Person #{lbph_id} ({lbph_conf:.0f})"
                color        = (0, 255, 0)
                cooldown_key = lbph_id                           # unique per known person
            else:
                label        = "Unknown"
                color        = (0, 100, 255)
                cooldown_key = f"unknown_{x // 100}_{y // 100}" # unique per screen region

            cv2.putText(frame, label, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            with active_keys_lock:
                already_active = cooldown_key in active_keys

            if already_active:
                cv2.putText(frame, "Processing...", (x, max(0, y - 28)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
                continue

            last_time = last_triggered.get(cooldown_key, 0)
            if now - last_time > REDETECT_COOLDOWN:
                last_triggered[cooldown_key] = now
                with active_keys_lock:
                    active_keys.add(cooldown_key)

                gc = gray_crop.copy()
                cc = color_crop.copy()
                ff = frame.copy()  # full frame for AWS

                def _run(g=gc, c=cc, f=ff, loc=location, kp=known_persons, ck=cooldown_key):
                    handle_face(g, c, loc, kp, ck, f)

                threading.Thread(target=_run, daemon=True).start()

        cv2.imshow("Face Memory System  -  press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    release_camera()
    cv2.destroyAllWindows()
    pygame.quit()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
