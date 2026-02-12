import os
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image

# --- 1. SILENCE TENSORFLOW WARNINGS ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import tensorflow as tf

# --- 2. FAIL-SAFE IMPORTS ---
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from streamlit_js_eval import streamlit_js_eval
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False

import streamlit as st

# --- 3. CLASS NAMES LIST ---
class_names = [
    'Apple Scab', 'Apple Black rot', 'Apple Cedar rust', 'Apple healthy',
    'Blueberry healthy', 'Cherry Powdery mildew', 'Cherry healthy',
    'Corn Gray leaf spot', 'Corn Common rust', 'Corn Northern Blight', 'Corn healthy', 
    'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 
    'Orange Citrus greening', 'Peach Bacterial spot', 'Peach healthy', 
    'Pepper Bacterial spot', 'Pepper healthy', 'Potato Early blight', 
    'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy', 
    'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy', 
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 
    'Tomato Leaf Mold', 'Tomato Septoria spot', 'Tomato Spider mites', 
    'Tomato Target Spot', 'Tomato Yellow Leaf Curl', 'Tomato mosaic virus', 'Tomato healthy'
]

# --- 4. LOCALIZATION TEXTS ---
texts = {
    'title': "🌱 Digital Agro-Health AI / ዲጂታል ሕርሻ ጥዕና",
    'instr': "Instructor: Ms. Winta Kidanemariam Hagos",
    'loc_label': "📍 Location / ቦታ:",
    'up_label': "Upload or Capture Leaf / ስእሊ የእትዉ (JPG, PNG)",
    'btn': "RUN ANALYSIS / መርምር",
    'diag': "Diagnosis / ውፅኢት:",
    'conf': "Confidence / እምነት:",
    'treat': "🛠️ Recommended Actions / ዝወሃቡ መፍትሒታት",
    'gps_err': "GPS logic loading... Please allow location access in your browser.",
    'cv_err': "OpenCV missing. Using standard image processing."
}

# --- 5. UI CONFIG ---
st.set_page_config(page_title="Agro-Health AI", page_icon="🌱", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 10px; background-color: #2e7d32; color: white; height: 3.5em; font-weight: bold; }
    .report-box { padding: 15px; border-radius: 10px; background-color: #f0f4f0; border-left: 5px solid #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

# --- 6. HELPER FUNCTIONS ---
def save_to_history(label, confidence, latitude, longitude):
    history_file = 'history.csv'
    new_data = pd.DataFrame({
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Diagnosis': [label],
        'Confidence': [f"{confidence:.2f}%"],
        'Lat': [latitude],
        'Lon': [longitude]
    })
    if not os.path.isfile(history_file):
        new_data.to_csv(history_file, index=False)
    else:
        new_data.to_csv(history_file, mode='a', header=False, index=False)

def clean_background(pil_image):
    if not OPENCV_AVAILABLE:
        return np.array(pil_image)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([10, 20, 20]), np.array([95, 255, 255]))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(img, img, mask=mask)
    result[mask == 0] = [128, 128, 128] # Gray padding
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

@st.cache_resource
def load_model():
    # Architecture rebuild for weight compatibility
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(38, activation='softmax')(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)
    try:
        model.load_weights('plant_model.weights.h5')
        return model
    except:
        return None

# Initialize Model & Data
model = load_model()
try:
    treatments_db = pd.read_csv('treatments.csv').set_index('disease_name').to_dict('index')
except:
    treatments_db = {}

# --- 7. SIDEBAR (GPS & HISTORY) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/2/20/Aksum_University_Logo.png", width=100)
    st.header("🌍 Research Tracker")
    
    # GPS Logic
    lat, lon = "Unknown", "Unknown"
    if GPS_AVAILABLE:
        loc = streamlit_js_eval(js_expressions="target.getLocation()", key="location")
        if loc:
            lat, lon = loc['coords']['latitude'], loc['coords']['longitude']
            st.success(f"📍 GPS Active: {lat:.4f}, {lon:.4f}")
        else:
            st.info(texts['gps_err'])
    
    st.write("---")
    if os.path.exists('history.csv'):
        hist_df = pd.read_csv('history.csv')
        st.subheader("Recent Logs")
        st.dataframe(hist_df.tail(5), hide_index=True)
        
        # Download Button for Researchers
        csv_data = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download All Data", csv_data, "agro_history.csv", "text/csv")

# --- 8. MAIN UI ---
st.title(texts['title'])
st.caption(texts['instr'])

col1, col2 = st.columns([1, 1])

with col1:
    file_input = st.file_uploader(texts['up_label'], type=["jpg","jpeg","png"])
    cam_input = st.camera_input("Take Photo / ብካሜራ ኣልዕል")
    input_source = file_input if file_input else cam_input

with col2:
    if input_source:
        raw_img = Image.open(input_source).convert('RGB')
        processed_img = clean_background(raw_img)
        
        st.image(processed_img, caption="AI Image Analysis View", use_container_width=True)
        
        if st.button(texts['btn']):
            if model is None:
                st.error("Model file not found! Please ensure 'plant_model.weights.h5' is in the folder.")
            else:
                with st.spinner("Processing..."):
                    # Preprocessing
                    img_resized = Image.fromarray(processed_img).resize((224, 224))
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_resized))
                    
                    # Prediction with Temperature Scaling (T=0.5)
                    preds = model.predict(np.expand_dims(img_array, axis=0))
                    exp_preds = np.exp(preds / 0.5)
                    probs = exp_preds / np.sum(exp_preds)
                    
                    idx = np.argmax(probs)
                    conf = np.max(probs) * 100
                    label = class_names[idx]

                    # Save to Local CSV
                    save_to_history(label, conf, lat, lon)

                    # Display Results
                    st.markdown(f"""
                    <div class="report-box">
                        <h3>{texts['diag']} {label}</h3>
                        <p>{texts['conf']} <b>{conf:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(conf))

                    # Treatment Advice
                    if label in treatments_db:
                        st.write("---")
                        st.subheader(texts['treat'])
                        st.write(f"**English:** {treatments_db[label]['en_treatment']}")
                        st.success(f"**Tigrinya:** {treatments_db[label]['tg_treatment']}")
                    else:
                        st.info("Additional treatment details are being updated by the Agricultural Office.")
