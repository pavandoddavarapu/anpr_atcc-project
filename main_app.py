import streamlit as st
import json
import cv2
import torch
from ultralytics import YOLO 
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime
from PIL import Image
import tempfile
import pandas as pd
import io
import matplotlib.pyplot as plt
import base64
import logging
import time
from dotenv import load_dotenv

# --- System Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("system_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- PyTorch 2.6+ Weights-Only Fix (Robust) ---
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ----------------------------------------------

# --- Configuration & Environment ---
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# OCR Settings
import sys
if sys.platform.startswith('win'):
    default_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    default_tesseract = "tesseract"

TESSERACT_CMD_PATH = os.getenv("TESSERACT_CMD_PATH", default_tesseract)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

# Model paths
LP_CUSTOM_WEIGHTS_PATH = os.getenv("LP_MODEL_PATH", "weights/best.pt")
ATCC_MODEL_PATH = os.getenv("ATCC_MODEL_PATH", "yolo11n.pt")

# Databases
LP_DB_PATH = os.getenv("LP_DB_PATH", "licensePlatesDatabase.db")
ATCC_DB_PATH = os.getenv("ATCC_DB_PATH", "traffic_analysis.db")

# NOTE: The TrafficDB class definition is assumed to be in 'traffic_db.py'
# The user's prompt did not include this file, so a placeholder class is used
# to prevent execution errors, but *you must provide the actual implementation*
# of TrafficDB for the second mode to function correctly.
class TrafficDB:
    def __init__(self, db_name=None):
        self.db_name = db_name or ATCC_DB_PATH
        self.setup_traffic_database()

    def setup_traffic_database(self):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        source_type TEXT,
                        vehicle_class TEXT,
                        count INTEGER,
                        traffic_level TEXT
                    )
                ''')
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"ATCC DB Setup Error: {e}")

    def save_result(self, timestamp, source_type, vehicle_class, count, traffic_level):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO analysis_results 
                    (timestamp, source_type, vehicle_class, count, traffic_level)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, source_type, vehicle_class, count, traffic_level))
                conn.commit()
                logger.info(f"Saved ATCC result: {vehicle_class} x {count}")
        except sqlite3.Error as e:
            logger.error(f"ATCC DB Save Error: {e}")

    def fetch_all_data(self):
        try:
            with sqlite3.connect(self.db_name) as conn:
                df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
                return df
        except sqlite3.Error as e:
            logger.error(f"ATCC DB Fetch Error: {e}")
            return pd.DataFrame()

    def clear_db(self):
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM analysis_results')
                conn.commit()
                logger.info("ATCC Database cleared.")
        except sqlite3.Error as e:
            logger.error(f"ATCC DB Clear Error: {e}")

# Class Names for LP Detector
LP_CLASS_NAMES = ["licence", "licenseplate"] 

# Check Tesseract availability once (relying on system PATH)
try:
    pytesseract.image_to_string(Image.new('RGB', (10, 10)), config='--psm 10')
    TESSERACT_AVAILABLE = True
    logger.info("Tesseract is available.")
except Exception as e:
    TESSERACT_AVAILABLE = False
    logger.warning(f"Tesseract is NOT available: {e}")
    
# --- Common/Cached YOLO Model Loader ---

@st.cache_resource
def initialize_yolo_model(weights_path):
    """Initializes and caches the YOLO model."""
    try:
        if not os.path.exists(weights_path):
            st.error(f"YOLO model not found at path: {weights_path}")
            return None
        model = YOLO(weights_path)
        return model
    except Exception as e:
        st.error(f"An error occurred during model loading from {weights_path}: {e}")
        return None

# --- DATABASE SETUP (License Plate) ---

def setup_license_plate_database():
    """Sets up the SQLite database and table for License Plates."""
    try:
        with sqlite3.connect(LP_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS LicensePlates (
                    id INTEGER PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    license_plate TEXT
                )
            ''')
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"LP DB Setup Error: {e}")

setup_license_plate_database()

# --- LICENSE PLATE MODE FUNCTIONS ---

def is_valid_license_plate(text):
    """Validates if the text looks like a real license plate."""
    if not text: return False
    if len(text) < 4 or len(text) > 10: return False
    if not text.isalnum(): return False
    has_alpha = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    if not (has_alpha and has_digit): return False
    return True

def has_significant_change(prev_frame, curr_frame, threshold=25, min_changed_pixels_ratio=0.01):
    """Detects if there is significant motion between two frames."""
    if prev_frame is None or curr_frame is None:
        return True
    
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_prev, gray_curr)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    non_zero_count = np.count_nonzero(thresh)
    total_pixels = thresh.size
    
    if total_pixels == 0: return True
    
    ratio = non_zero_count / total_pixels
    return ratio >= min_changed_pixels_ratio

def tesseract_ocr_process(frame, x1, y1, x2, y2):
    """Performs Tesseract OCR on a cropped license plate."""
    if not TESSERACT_AVAILABLE:
        return "OCR_PATH_ERROR"
        
    h, w, _ = frame.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return "INVALID_CROP" 
        
    cropped_frame = frame[y1:y2, x1:x2].copy()
    
    try:
        # Pre-processing: Grayscale -> CLAHE (Night Vision) -> Threshold -> Blur
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.medianBlur(thresh, 3) 
        
        pil_image = Image.fromarray(thresh)
        
        # PSM 7: Treat as a single text line/block (ideal for plates)
        ocr_config = '--psm 7 -l eng'
        raw_text = pytesseract.image_to_string(pil_image, config=ocr_config)
    except Exception:
        return "OCR_EXEC_FAIL"
    
    # --- Cleanup and Formatting ---
    pattern = re.compile(r'[^A-Z0-9\s]')
    cleaned_text = pattern.sub('', raw_text.upper()).strip()
    final_text = cleaned_text.replace(" ", "") 

    if not is_valid_license_plate(final_text):
        return None
        
    return final_text

def save_lp_data(license_plates, startTime, endTime):
    """Saves license plate data to the database."""
    if not license_plates:
        return

    save_to_lp_database(license_plates, startTime, endTime)


def save_to_lp_database(license_plates, start_time, end_time):
    """Saves license plate data to the SQLite database (LicensePlates table)."""
    try:
        with sqlite3.connect(LP_DB_PATH) as conn:
            cursor = conn.cursor()
            for plate in license_plates:
                cursor.execute('''
                    INSERT INTO LicensePlates(start_time, end_time, license_plate)
                    VALUES (?, ?, ?)
                ''', (start_time.isoformat(), end_time.isoformat(), plate))
            conn.commit()
            logger.info(f"Saved {len(license_plates)} license plates to DB.")
    except sqlite3.Error as e:
        logger.error(f"LP DB Save Error: {e}")

def process_lp_frame(frame, license_plates_dict, model):
    """Runs YOLO detection and Tesseract OCR on a single frame."""
    if model is None:
        return frame
        
    results = model.predict(frame, conf=0.45, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            conf = math.ceil(box.conf[0].item() * 100) / 100
            classNameInt = int(box.cls[0].item())
            
            if classNameInt < len(LP_CLASS_NAMES):
                clsName = LP_CLASS_NAMES[classNameInt]
            else:
                clsName = "Unknown" 

            # Execute OCR function
            label = tesseract_ocr_process(frame.copy(), x1, y1, x2, y2)
            
            if label:
                if label not in license_plates_dict:
                    # Capture the vehicle context crop and encode to base64
                    h, w, _ = frame.shape
                    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    
                    if cx2 > cx1 and cy2 > cy1:
                        plate_width = cx2 - cx1
                        plate_height = cy2 - cy1
                        
                        # Expand dimensions for context (vehicle)
                        vx1 = max(0, cx1 - int(2.5 * plate_width))
                        vx2 = min(w, cx2 + int(2.5 * plate_width))
                        vy1 = max(0, cy1 - int(4.0 * plate_height))
                        vy2 = min(h, cy2 + int(1.5 * plate_height))
                        
                        if vx2 > vx1 and vy2 > vy1:
                            vehicle_crop = frame[vy1:vy2, vx1:vx2].copy()
                            # Draw a box around the plate INSIDE the vehicle crop for clarity
                            cv2.rectangle(vehicle_crop, (cx1 - vx1, cy1 - vy1), (cx2 - vx1, cy2 - vy1), (0, 255, 0), 2)
                            _, buffer = cv2.imencode('.jpg', vehicle_crop)
                        else:
                            # Fallback to basic crop
                            cropped_plate = frame[cy1:cy2, cx1:cx2].copy()
                            _, buffer = cv2.imencode('.jpg', cropped_plate)
                            
                        b64_str = base64.b64encode(buffer).decode('utf-8')
                        license_plates_dict[label] = f"data:image/jpeg;base64,{b64_str}"
                    else:
                        license_plates_dict[label] = None
                display_label = label
            else:
                display_label = f'{clsName}:{conf:.2f}'

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw text background
            textSize = cv2.getTextSize(display_label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + textSize[0] + 5, y1 - textSize[1] - 8
            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, display_label, (x1, y1 - 4), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    return frame

def lp_video_processing_loop(cap, model, watchlist=None):
    """Processes video from a capture object (file or camera) for License Plate Detection."""
    st.subheader("Processing Video Feed... 🚗")
    
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    plate_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    startTime = datetime.now()
    license_plates = {}
    alerts_triggered = set()
    frame_count = 0
    session_data = []
    last_processed_frame = None
    
    is_file = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 
    max_frames = 600 if not is_file else cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret or frame is None:
            break

        frame_count += 1
        
        h, w, _ = frame.shape
        if w > 800:
            frame = cv2.resize(frame, (800, int(800 * h / w)))

        # Dynamic frame skip based on motion
        if not has_significant_change(last_processed_frame, frame):
            # We still need to respect the 600 frame limit for webcam
            if not is_file and frame_count >= 600:
                break 
            continue
            
        last_processed_frame = frame.copy()
        
        processed_frame = process_lp_frame(frame, license_plates, model)
        
        # Check Watchlist Alerts
        if watchlist:
            for plate in license_plates.keys():
                if plate in watchlist and plate not in alerts_triggered:
                    alert_placeholder.error(f"🚨 **WATCHLIST ALERT!** Wanted vehicle `{plate}` has been detected!")
                    alerts_triggered.add(plate)
                    
        frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"Frame {frame_count}/{int(max_frames) if is_file else 'live'}")

        # Time-based saving logic (every 20 seconds)
        currentTime = datetime.now()
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            save_lp_data(list(license_plates.keys()), startTime, endTime)
            for plate, img_uri in license_plates.items():
                session_data.append({"Start Time": startTime.strftime('%Y-%m-%d %H:%M:%S'), "End Time": endTime.strftime('%Y-%m-%d %H:%M:%S'), "License Plate": plate, "Image": img_uri})
            startTime = currentTime
            license_plates.clear()

        status_placeholder.text(f"Frames processed: {frame_count} | Unique Entries: {len(license_plates)} (since last save)")
        plate_placeholder.json({"Detected Entries (since last save)": list(license_plates.keys())})
        
        if not is_file and frame_count >= 600:
             break 

        cv2.waitKey(1) 

    if license_plates:
        endTime = datetime.now()
        save_lp_data(list(license_plates.keys()), startTime, endTime)
        for plate, img_uri in license_plates.items():
            session_data.append({"Start Time": startTime.strftime('%Y-%m-%d %H:%M:%S'), "End Time": endTime.strftime('%Y-%m-%d %H:%M:%S'), "License Plate": plate, "Image": img_uri})
        
    cap.release()
    frame_placeholder.empty()
    status_placeholder.empty()
    plate_placeholder.empty()
    st.success("Video processing finished.")

    if session_data:
        st.subheader("Detected License Plates")
        
        df = pd.DataFrame(session_data)
        
        col_search, col_export = st.columns([3, 1])
        with col_search:
            search_term = st.text_input("🔍 Search Plates:", "", key="video_search")
        with col_export:
            st.markdown("<br>", unsafe_allow_html=True)
            csv = df.drop(columns=["Image"], errors="ignore").to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Export to CSV",
                data=csv,
                file_name=f"anpr_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime='text/csv',
                key="video_export"
            )
            
        if search_term:
            df = df[df['License Plate'].str.contains(search_term.upper(), na=False)]
            
        st.dataframe(
            df,
            column_config={"Image": st.column_config.ImageColumn("Plate Image")},
            use_container_width=True
        )

# --- ATCC (VEHICLE ANALYZER) MODE FUNCTIONS ---

def calculate_traffic_level(total_count):
    """Classifies traffic density based on total vehicle count."""
    if total_count == 0:
        return "No Traffic"
    elif total_count <= 5:
        return "Low Traffic"
    elif total_count <= 15:
        return "Medium Traffic"
    else:
        return "High Traffic"

def process_atcc_detection(results, db: TrafficDB, source_type="Image/Video"):
    """
    Processes YOLO detection results, logs to DB, and returns summary data.
    Assumes `results` is an ultralytics 'Results' object or a list of such objects.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not isinstance(results, list):
        results = [results]

    total_vehicles = 0
    class_counts = {}

    for res in results:
        if hasattr(res.boxes, 'cls') and res.boxes.cls is not None:
            detection_classes = res.boxes.cls.cpu().numpy()
            
            try:
                class_names = [results[0].names[int(cls_id)] for cls_id in detection_classes]
            except (AttributeError, KeyError):
                class_names = [f"Class {int(cls_id)}" for cls_id in detection_classes]

            for class_name in class_names:
                total_vehicles += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    traffic_level = calculate_traffic_level(total_vehicles)

    for vehicle_class, count in class_counts.items():
        db.save_result(timestamp, source_type, vehicle_class, count, traffic_level)
    
    if not class_counts:
          db.save_result(timestamp, source_type, "N/A", 0, "No Traffic")


    summary = {
        'timestamp': timestamp,
        'total_vehicles': total_vehicles,
        'traffic_level': traffic_level,
        'class_counts': class_counts
    }
    return summary

def annotate_atcc_image(result):
    """Annotates a single YOLO result image and returns it as a PNG byte buffer."""
    annotated_img = result.plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    buf = io.BytesIO()
    plt.figure(figsize=(8, 8))
    plt.imshow(annotated_img_rgb)
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return buf

def display_raw_data(db: TrafficDB):
    """Displays the raw contents of the SQLite database in a new expander."""
    with st.expander("Raw Database Table (analysis_results)", expanded=True):
        raw_df = db.fetch_all_data()
        st.dataframe(raw_df, use_container_width=True)
        st.markdown(f"Total Records: **{len(raw_df)}**")

# --- CUSTOM CSS STYLES (FROM ATCC APP) ---

def apply_custom_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #3B82F6; /* Blue-500 */
            text-align: center;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #60A5FA; /* Blue-400 */
        }
        .stButton>button {
            background-color: #10B981; /* Emerald-500 */
            color: white;
            font-weight: 600;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            transition: background-color 0.2s;
        }
        .stButton>button:hover {
            background-color: #059669; /* Emerald-600 */
        }
        .traffic-low { color: #10B981; font-weight: 700; }
        .traffic-medium { color: #F59E0B; font-weight: 700; }
        .traffic-high { color: #EF4444; font-weight: 700; }
        .traffic-none { color: #6B7280; font-weight: 700; }
        .stFileUploader {
            border: 2px dashed #9CA3AF;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        .caption {
            font-size: 0.8rem;
            color: #6B7280;
            text-align: center;
            margin-top: -0.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


# --- LICENSE PLATE MODE LAYOUT ---

def license_plate_mode(model):
    st.title("License Plate Detector & Tesseract OCR 🏷️")
    st.sidebar.title("LP Detector Options")

    # Display Tesseract status
    if TESSERACT_AVAILABLE:
        st.sidebar.success("Tesseract OCR Active.")
    else:
        st.sidebar.error("Tesseract Error: Using placeholders for OCR results.")
        st.warning("Tesseract is unavailable. Please ensure it's installed and added to your system PATH.")
    
    if model is None:
        st.error("License Plate YOLO model did not load. Detection is disabled.")
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Watchlist Settings")
    watchlist_input = st.sidebar.text_area("Enter plates to track (comma separated)", placeholder="e.g. MH01AB1234, DL4CAF4943")
    watchlist = [p.strip().upper() for p in watchlist_input.split(',') if p.strip()]
    
    st.sidebar.markdown("---")
    
    source_option = st.sidebar.radio(
        "Select Input Source:",
        ('Upload Video', 'Upload Photo', 'Use Webcam (Experimental)')
    )
    
    st.markdown("---")

    # --- 1. Video Upload ---
    if source_option == 'Upload Video':
        st.header("Video File Upload")
        uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1] or ".mp4") as tfile:
                tfile.write(uploaded_file.read())
                temp_video_path = tfile.name
                
            if st.button("Start Processing Video 🎬"):
                with st.spinner('Initializing video stream...'):
                    cap = cv2.VideoCapture(temp_video_path)
                lp_video_processing_loop(cap, model, watchlist)
                
            if os.path.exists(temp_video_path):
                try:
                    time.sleep(1)
                    os.unlink(temp_video_path)
                except PermissionError:
                    st.warning("File still in use, skipping delete.")
                except Exception as e:
                    st.warning(f"Could not clean up temporary file: {e}")

    # --- 2. Photo Upload ---
    elif source_option == 'Upload Photo':
        st.header("Image File Upload")
        uploaded_image = st.file_uploader("Choose a photo...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert('RGB')
            img_array = np.array(image)
            frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption='Original Image', use_container_width=True)
            
            with col2:
                if st.button("Analyze Photo 🖼️"):
                    with st.spinner('Analyzing image...'):
                        license_plates = {}
                        h, w, _ = frame.shape
                        if w > 800:
                            frame = cv2.resize(frame, (800, int(800 * h / w)))
                        
                        processed_frame = process_lp_frame(frame, license_plates, model)
                        
                        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption='Processed Image', use_container_width=True)
                        
                    if license_plates:
                        st.success("Analysis Complete! Detected entries saved to DB.")
                        
                        # Watchlist Check
                        for plate in license_plates.keys():
                            if watchlist and plate in watchlist:
                                st.error(f"🚨 **WATCHLIST ALERT!** Wanted vehicle `{plate}` has been detected!")
                                
                        current_time = datetime.now()
                        save_lp_data(list(license_plates.keys()), current_time, current_time)
                        
                        session_data = [{"Time": current_time.strftime('%Y-%m-%d %H:%M:%S'), "License Plate": plate, "Image": img_uri} for plate, img_uri in license_plates.items()]
                        st.subheader("Detected License Plates")
                        
                        df = pd.DataFrame(session_data)
                        
                        col_search, col_export = st.columns([3, 1])
                        with col_search:
                            search_term = st.text_input("🔍 Search Plates:", "", key="photo_search")
                        with col_export:
                            st.markdown("<br>", unsafe_allow_html=True)
                            csv = df.drop(columns=["Image"], errors="ignore").to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Export to CSV",
                                data=csv,
                                file_name=f"anpr_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime='text/csv',
                                key="photo_export"
                            )
                            
                        if search_term:
                            df = df[df['License Plate'].str.contains(search_term.upper(), na=False)]
                            
                        st.dataframe(
                            df,
                            column_config={"Image": st.column_config.ImageColumn("Plate Image")},
                            use_container_width=True
                        )
                    else:
                        st.info("No license plate objects were detected by YOLO.")


    # --- 3. Webcam Input ---
    elif source_option == 'Use Webcam (Experimental)':
        st.header("Webcam Input (Experimental)")
        st.warning("Webcam capture can be inconsistent in Streamlit. This mode will attempt to run for ~600 frames.")
        
        if st.button("Start Camera 📸"):
            with st.spinner('Attempting to open camera...'):
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open camera. Check permissions or if another application is using it.")
            else:
                lp_video_processing_loop(cap, model, watchlist)

# --- ATCC MODE LAYOUT ---

def atcc_mode(model, db: TrafficDB):
    
    apply_custom_styles()
    st.markdown('<div class="main-header">ATCC YOLOv11 Vehicle Analyzer</div>', unsafe_allow_html=True)
    st.markdown("---")

    if model is None:
        st.error("ATCC YOLO model did not load. Detection is disabled.")
        return

    # --- Sidebar for Settings ---
    with st.sidebar:
        st.header("ATCC Analyzer Options")
        analysis_mode = st.radio("Choose Input Source",
            ('Upload Image/Video', 'Webcam Capture'),
            index=0,
            key='atcc_analysis_mode')
        
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
        
        st.markdown("---")
        st.subheader("Model Information")
        st.markdown(f"Model: `{ATCC_MODEL_PATH}`")
        class_names = model.names.values() if hasattr(model, 'names') else ["(Classes not loaded)"]
        st.markdown(f"Classes Detected ({len(class_names)}): {', '.join(class_names)}")
        st.markdown("---")
        st.markdown("### Data Storage")
        
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            if st.button("View Raw DB"):
                st.session_state['view_raw_db'] = True
        with col_db2:
            if st.button("Clear DB"):
                db.clear_db()
                st.success("Database cleared! Reloading...")
                st.rerun() 

        if st.session_state.get('view_raw_db', False):
            st.session_state['view_raw_db'] = False
            display_raw_data(db)

    # --- Main Content Area ---
    col_input, col_results = st.columns([1, 1])

    uploaded_file = None
    process_button = False

    with col_input:
        st.subheader("Input Source")
        
        if analysis_mode == 'Upload Image/Video':
            uploaded_file = st.file_uploader(
                "Upload an Image (jpg, png) or Video (mp4, mov, avi)",
                type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'],
                key='atcc_uploader'
            )
            process_button = st.button("Start Analysis (Upload)", key='atcc_process_upload')

        elif analysis_mode == 'Webcam Capture':
            st.warning("Webcam analysis is resource-intensive.")
            webcam_image = st.camera_input("Capture an image or video segment.", key='atcc_webcam')
            
            uploaded_file = webcam_image
            process_button = st.button("Start Analysis (Webcam)", disabled=not webcam_image, key='atcc_process_webcam')
        
        media_container = st.container()
        if uploaded_file and analysis_mode == 'Upload Image/Video':
            media_type = uploaded_file.type.split('/')[0]
            if media_type == 'image':
                media_container.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            elif media_type == 'video':
                media_container.markdown('<p class="caption">Uploaded Video</p>', unsafe_allow_html=True)
                media_container.video(uploaded_file)

    # --- Analysis Execution ---
    results_summary = None
    annotated_media = None
    
    if uploaded_file and process_button:
        
        media_type = uploaded_file.type.split('/')[0]
        
        with col_results:
            st.subheader("Detection Results")
            progress_bar = st.progress(0)
            
            temp_path = None
            try:
                # --- Temporary File Handling ---
                file_extension = os.path.splitext(uploaded_file.name)[1] if uploaded_file.name else f".{uploaded_file.type.split('/')[1]}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file.flush()
                    temp_path = tmp_file.name

                args = {
                    'conf': confidence_threshold,
                    'iou': iou_threshold,
                    'save': False,
                    'verbose': False
                }
                
                # --- Run Detection ---
                if media_type == 'image' or (analysis_mode == 'Webcam Capture' and uploaded_file):
                    st.info("Processing image...")
                    results = model.predict(temp_path, **args)
                    annotated_media_buffer = annotate_atcc_image(results[0])
                    results_summary = process_atcc_detection(results, db, source_type="Image" if media_type == 'image' else "Webcam Snapshot")
                    
                    st.markdown('<p class="caption">Annotated Image</p>', unsafe_allow_html=True)
                    st.image(annotated_media_buffer, use_container_width=True)
                    progress_bar.progress(100)
                    
                elif media_type == 'video' and analysis_mode == 'Upload Image/Video':
                    st.info("Processing video feed...")
                    
                    cap = cv2.VideoCapture(temp_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames == 0: total_frames = 100

                    frame_idx = 0
                    video_placeholder = st.empty()
                    last_annotated_frame = None
                    last_processed_frame = None
                    
                    track_history = {}
                    unique_vehicles_classes = {}
                    incoming_count = 0
                    outgoing_count = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        # Use model.track for assigning persistent IDs across frames
                        results = model.track(frame, persist=True, **args)
                        annotated_frame = results[0].plot()
                        
                        # Draw Virtual Line (halfway down the frame)
                        line_y = int(frame_height / 2)
                        cv2.line(annotated_frame, (0, line_y), (frame_width, line_y), (0, 0, 255), 3)
                        
                        # Directional Tracking Logic
                        if results[0].boxes is not None and results[0].boxes.id is not None:
                            boxes = results[0].boxes.xywh.cpu()
                            track_ids = results[0].boxes.id.int().cpu().tolist()
                            cls_ids = results[0].boxes.cls.int().cpu().tolist()
                            
                            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                                x, y, w, h = box
                                
                                if track_id not in unique_vehicles_classes:
                                    try:
                                        cls_name = results[0].names[cls_id]
                                    except (AttributeError, KeyError):
                                        cls_name = f"Class {cls_id}"
                                    unique_vehicles_classes[track_id] = cls_name
                                
                                if track_id in track_history:
                                    prev_y = track_history[track_id]
                                    # Vehicle crossed the virtual line
                                    if prev_y < line_y and y >= line_y:
                                        incoming_count += 1
                                    elif prev_y > line_y and y <= line_y:
                                        outgoing_count += 1
                                
                                track_history[track_id] = y
                        
                        # Overlay counting UI
                        cv2.putText(annotated_frame, f"Incoming: {incoming_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        cv2.putText(annotated_frame, f"Outgoing: {outgoing_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                        
                        last_annotated_frame = annotated_frame
                        last_processed_frame = frame.copy()
                        
                        # Write to video (keeps original length/speed by reusing frames)
                        out.write(last_annotated_frame)
                        
                        # Update Streamlit UI less frequently (every ~15 frames) to avoid websocket bottleneck
                        if frame_idx % 15 == 0:
                            video_placeholder.image(last_annotated_frame, channels="BGR", caption=f"Processing Frame {frame_idx + 1}/{total_frames}", use_container_width=True)

                        frame_idx += 1
                        progress_bar.progress(min(int(frame_idx / total_frames * 100), 100))

                    cap.release()
                    out.release()
                    
                    class_counts = {}
                    for cls_name in unique_vehicles_classes.values():
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    total_vehicles = sum(class_counts.values())
                    traffic_level = calculate_traffic_level(total_vehicles)

                    for vehicle_class, count in class_counts.items():
                        db.save_result(timestamp, "Video", vehicle_class, count, traffic_level)
                    
                    if not class_counts:
                        db.save_result(timestamp, "Video", "N/A", 0, "No Traffic")

                    results_summary = {
                        'timestamp': timestamp,
                        'total_vehicles': total_vehicles,
                        'traffic_level': traffic_level,
                        'class_counts': class_counts
                    }
                    annotated_media = out_path
                    
                    video_placeholder.empty()
                    st.markdown('<p class="caption">Annotated Video Output</p>', unsafe_allow_html=True)
                    st.video(annotated_media, format="video/mp4", start_time=0)
                    
            except Exception as e:
                st.error(f"An error occurred during detection: {e}")
                results_summary = None
            finally:
                # Ensure resources are released
                if 'cap' in locals() and cap is not None and cap.isOpened():
                    cap.release()
                if 'out' in locals() and out is not None:
                    out.release()
                    
                # Clean up temporary files
                if temp_path and os.path.exists(temp_path):
                    try:
                        time.sleep(1)
                        os.remove(temp_path)
                    except PermissionError:
                        logger.warning("Temp input file still in use, skipping delete.")
                    except Exception as e:
                        logger.warning(f"Could not remove temp input file: {e}")
                
                if annotated_media and os.path.exists(annotated_media) and media_type == 'video':
                    try:
                        time.sleep(1)
                        os.remove(annotated_media)
                    except PermissionError:
                        logger.warning("Temp output video still in use, skipping delete.")
                    except Exception as e:
                        logger.warning(f"Could not remove temp output video: {e}")
                    
                progress_bar.empty()

    # --- Display Results and Visualization ---
    st.markdown("---")
    
    if results_summary:
        st.subheader("Analysis Summary")
        
        traffic_level_class = results_summary['traffic_level'].lower().replace(' ', '-')
        
        st.markdown(f"""
            <div style="background-color: #F3F4F6; padding: 15px; border-radius: 0.5rem;">
                <p style="font-size: 1.1rem; margin-bottom: 5px;">
                    <strong>Timestamp:</strong> {results_summary['timestamp']}
                </p>
                <p style="font-size: 1.1rem; margin-bottom: 5px;">
                    <strong>Total Vehicles Detected:</strong> <span style="color:#3B82F6; font-weight:700;">{results_summary['total_vehicles']}</span>
                </p>
                <p style="font-size: 1.1rem; margin-bottom: 0;">
                    <strong>Traffic Level:</strong>
                    <span class="traffic-{traffic_level_class}">{results_summary['traffic_level']}</span>
                </p>
            </div>
            <br>
        """, unsafe_allow_html=True)

        st.subheader("Vehicle Breakdown")
        
        class_counts_dict = results_summary.get('class_counts', {})
        if class_counts_dict:
            class_df = pd.DataFrame(
                list(class_counts_dict.items()),
                columns=['Vehicle Class', 'Count']
            )
            class_df.set_index('Vehicle Class', inplace=True)
            st.table(class_df)
        else:
            st.info("No vehicles were detected in the media.")


    st.markdown("---")
    st.header("Traffic History Data")
    
    db_df = db.fetch_all_data()
    
    if not db_df.empty:
        st.subheader("Raw Analysis Log")
        st.dataframe(db_df, use_container_width=True)
        st.markdown(f"Total Records: **{len(db_df)}**")
        
        traffic_summary = db_df['traffic_level'].value_counts().reset_index()
        traffic_summary.columns = ['Traffic Level', 'Count']
        st.subheader("Traffic Level Totals")
        st.dataframe(traffic_summary)

    else:
        st.info("No historical analysis data is available yet. Run an analysis to populate the database.")


# --- ANALYTICS DASHBOARD MODE ---

def analytics_dashboard_mode(db):
    apply_custom_styles()
    st.markdown('<div class="main-header">Global Analytics Dashboard 📊</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Traffic Analysis Overview")
    with col2:
        if st.button("🧹 Purge Old Data (> 30 days)"):
            try:
                with sqlite3.connect(db.db_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM analysis_results WHERE date(timestamp) < date('now', '-30 days')")
                    conn.commit()
                
                with sqlite3.connect(LP_DB_PATH) as conn_lp:
                    cursor_lp = conn_lp.cursor()
                    cursor_lp.execute("DELETE FROM LicensePlates WHERE date(start_time) < date('now', '-30 days')")
                    conn_lp.commit()
                    
                st.success("Old data purged successfully!")
                logger.info("Purged old database records (>30 days).")
            except Exception as e:
                st.error(f"Error purging data: {e}")
                logger.error(f"Purge data error: {e}")
                
    # Load ATCC data
    df_atcc = db.fetch_all_data()
    
    # Load LP data
    try:
        with sqlite3.connect(LP_DB_PATH) as conn_lp:
            df_lp = pd.read_sql_query("SELECT * FROM LicensePlates", conn_lp)
    except sqlite3.Error as e:
        logger.error(f"Error loading LP data for dashboard: {e}")
        df_lp = pd.DataFrame()
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Vehicles Counted (ATCC)", df_atcc['count'].sum() if not df_atcc.empty else 0)
    col_b.metric("License Plates Logged", len(df_lp) if not df_lp.empty else 0)
    col_c.metric("ATCC Sessions", len(df_atcc['timestamp'].unique()) if not df_atcc.empty else 0)
    
    st.markdown("---")
    
    col_charts1, col_charts2 = st.columns(2)
    
    with col_charts1:
        st.subheader("Vehicle Class Distribution")
        if not df_atcc.empty:
            class_dist = df_atcc.groupby('vehicle_class')['count'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(class_dist['count'], labels=class_dist['vehicle_class'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("No ATCC data available.")
            
    with col_charts2:
        st.subheader("Traffic Volume Over Time")
        if not df_atcc.empty:
            df_atcc['timestamp'] = pd.to_datetime(df_atcc['timestamp'])
            time_vol = df_atcc.groupby('timestamp')['count'].sum().reset_index()
            st.line_chart(data=time_vol, x='timestamp', y='count', use_container_width=True)
        else:
            st.info("No ATCC data available.")

# --- MAIN APPLICATION ENTRY POINT ---

def main():
    st.set_page_config(page_title="Combined YOLO App", layout="wide", initial_sidebar_state="expanded")
    
    st.sidebar.title("App Selection")
    
    # Main selector for the three application modes
    app_mode = st.sidebar.radio(
        "Select Application Mode:",
        ('License Plate Detector (LP) / OCR', 'Vehicle Traffic Analyzer (ATCC)', 'Global Analytics Dashboard 📊'),
        key='app_mode_select'
    )
    
    st.sidebar.markdown("---")

    # Initialize DB (Traffic) in session state
    if 'atcc_db' not in st.session_state:
        st.session_state['atcc_db'] = TrafficDB()
    atcc_db = st.session_state['atcc_db']

    if app_mode == 'License Plate Detector (LP) / OCR':
        # Load the LP model
        lp_model = initialize_yolo_model(LP_CUSTOM_WEIGHTS_PATH)
        license_plate_mode(lp_model)
    elif app_mode == 'Vehicle Traffic Analyzer (ATCC)':
        # Load the ATCC model
        atcc_model = initialize_yolo_model(ATCC_MODEL_PATH)
        atcc_mode(atcc_model, atcc_db)
    elif app_mode == 'Global Analytics Dashboard 📊':
        analytics_dashboard_mode(atcc_db)

if __name__ == '__main__':
    main()