import streamlit as st
import joblib
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import json
import time

# Set page config
st.set_page_config(
    page_title="Lung Nodule Prediction Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful visual layout and modern aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@500;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-bottom: 1.8rem;
    }
    
    .card-container {
        background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.95));
        border: 1px solid rgba(14,165,233,0.25);
        border-radius: 18px;
        padding: 1.15rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16);
    }

    .hero-panel {
        background: linear-gradient(135deg, rgba(2,132,199,0.26), rgba(14,165,233,0.12));
        border: 1px solid rgba(125,211,252,0.24);
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #cbd5e1;
    }

    .metric-value-large {
        font-size: 2.4rem;
        font-weight: 700;
        color: #00f2fe;
    }

    .tiny-label {
        font-size: 0.74rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #7dd3fc;
        margin-bottom: 0.3rem;
    }

    .recommendation-card {
        background: linear-gradient(135deg, rgba(2,6,23,0.96), rgba(15,23,42,0.92));
        border: 1px solid rgba(34,211,238,0.22);
        border-radius: 20px;
        padding: 1.15rem 1.25rem;
        margin-top: 0.75rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
    }

    .recommendation-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .recommendation-summary {
        color: #e2e8f0;
        margin: 0.6rem 0 0.8rem;
        line-height: 1.55;
    }

    .recommendation-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.7rem;
        margin-bottom: 0.7rem;
    }

    .info-card {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    .highlight {
        font-weight: 700;
        color: #f8fafc;
    }

    .action-list {
        margin: 0.4rem 0 0.6rem 1rem;
        color: #e2e8f0;
        line-height: 1.55;
    }

    .subtle-text {
        color: #cbd5e1;
        font-size: 0.94rem;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.35rem;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0.7rem;
    }
    
    .status-badge-benign {
        background-color: rgba(72, 187, 120, 0.15);
        color: #48bb78;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        border: 1px solid rgba(72, 187, 120, 0.3);
        display: inline-block;
    }

    .status-badge-malignant {
        background-color: rgba(245, 101, 101, 0.15);
        color: #f56565;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        border: 1px solid rgba(245, 101, 101, 0.3);
        display: inline-block;
    }
    
    .model-pill {
        background-color: #2d3748;
        color: #e2e8f0;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions & Caching -----------------

@st.cache_resource
def load_models(rf_path, nb_path, gb_path=None):
    """Load Random Forest, Naive Bayes, and an optional Gradient Boosting model using joblib."""
    rf = joblib.load(rf_path)
    nb = joblib.load(nb_path)
    gb = joblib.load(gb_path) if gb_path and os.path.exists(gb_path) else None
    return rf, nb, gb

@st.cache_data
def load_coco_annotations(json_path):
    """Load COCO annotation file for validation metadata."""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading COCO annotations: {e}")
    return None

def preprocess_image(pil_image):
    """Preprocess image to match training input: RGB -> Resize 224x224 -> Rescale 1/255 -> Flatten."""
    img_rgb = pil_image.convert("RGB")
    img_resized = img_rgb.resize((224, 224))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_flat = img_arr.flatten().reshape(1, -1)
    return img_flat


def safe_roc_curve(y_true, y_scores):
    """Return ROC curve data only when both classes are present."""
    if len(np.unique(y_true)) < 2:
        return None
    return roc_curve(y_true, y_scores, pos_label=1)


def get_care_pathway(probability, pred_label):
    """Convert model probability into a clinician-friendly care pathway."""
    if pred_label == 1:
        if probability >= 0.80:
            return {
                "badge": "High concern",
                "badge_class": "status-badge-malignant",
                "priority": "Urgent specialist review",
                "summary": "The model indicates a strong likelihood of a malignant finding. This should trigger prompt specialist evaluation, biopsy confirmation, and staging before any treatment plan is finalized.",
                "actions": [
                    "Refer to a pulmonologist or oncologist for confirmation and next-step planning.",
                    "Arrange biopsy and staging studies such as CT or PET review as appropriate.",
                    "Discuss whether surgery, radiation, systemic therapy, or supportive care is appropriate after diagnosis."
                ],
                "treatments": ["Surgery", "Radiation therapy", "Chemotherapy", "Targeted therapy", "Immunotherapy", "Supportive care"],
            }
        elif probability >= 0.60:
            return {
                "badge": "Moderate concern",
                "badge_class": "status-badge-malignant",
                "priority": "Prompt clinician follow-up",
                "summary": "The scan pattern is concerning enough to merit specialist review and additional diagnostic workup.",
                "actions": [
                    "Book a follow-up consultation with a lung specialist.",
                    "Discuss whether repeat imaging or biopsy is needed based on symptoms and history.",
                    "Prepare for treatment discussions if pathology confirms malignancy."
                ],
                "treatments": ["Radiation therapy", "Chemotherapy", "Targeted therapy", "Immunotherapy"],
            }
        return {
            "badge": "Watchful monitoring",
            "badge_class": "status-badge-malignant",
            "priority": "Clinical review",
            "summary": "The model still flags risk, but the evidence is not yet strong enough to assume a diagnosis. More review is needed.",
            "actions": [
                "Continue medical follow-up and review symptoms carefully.",
                "Ask about repeat imaging or specialist consultation if symptoms change.",
                "Do not rely on home remedies or supplements as a substitute for evaluation."
            ],
            "treatments": ["Monitoring", "Repeat imaging", "Supportive care"],
        }

    return {
        "badge": "Lower concern",
        "badge_class": "status-badge-benign",
        "priority": "Routine follow-up",
        "summary": "The model does not indicate a strong malignant pattern, but this does not replace a clinician's judgment.",
        "actions": [
            "Continue routine follow-up and discuss symptoms with a physician.",
            "Maintain healthy lung habits and stop smoking if applicable.",
            "Seek medical advice promptly if new symptoms such as cough, chest pain, or weight loss appear."
        ],
        "treatments": ["Monitoring", "Symptom management", "Preventive care"],
    }


def render_patient_drug_guide(probability, pred_label):
    """Show a modern, patient-friendly treatment overview based on the model result."""
    st.markdown("---")
    st.markdown('<div class="section-title">🧭 Model-Guided Care Pathway</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Clinical guidance derived from the current prediction confidence.</div>', unsafe_allow_html=True)
    pathway = get_care_pathway(probability, pred_label)
    st.markdown(
        f"""
        <div class="recommendation-card">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:0.8rem; flex-wrap:wrap;">
                <div>
                    <div class="tiny-label">Care pathway</div>
                    <div class="recommendation-title">{pathway['badge']}</div>
                </div>
                <span class="{pathway['badge_class']}">{pathway['badge']}</span>
            </div>
            <p class="recommendation-summary">{pathway['summary']}</p>
            <div class="recommendation-grid">
                <div class="info-card">
                    <div class="tiny-label">Priority</div>
                    <div class="highlight">{pathway['priority']}</div>
                </div>
                <div class="info-card">
                    <div class="tiny-label">Possible treatment categories</div>
                    <div class="highlight">{', '.join(pathway['treatments'])}</div>
                </div>
            </div>
            <ul class="action-list">
                {''.join([f'<li>{action}</li>' for action in pathway['actions']])}
            </ul>
            <p class="subtle-text"><strong>Important:</strong> This is not a prescription and does not replace medical review. Only a qualified clinician can decide the correct treatment after biopsy, imaging, and clinical assessment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- App Layout & Sidebar -----------------

st.markdown('<div class="hero-panel"><div class="hero-title">Lung Nodule Care Intelligence</div><div class="hero-subtitle">Modern scan review, model-driven insight, and patient-friendly treatment guidance in one dashboard.</div></div>', unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/color/96/000000/lungs.png", width=80)
st.sidebar.header("📁 Configuration")

# Path variables
default_model_dir = os.path.dirname(os.path.abspath(__file__))
default_validation_dir = os.path.join(default_model_dir, "lung_ct_version_n_512.v2i.coco", "valid")

# Sidebar controls for paths
model_dir = st.sidebar.text_input("Model Directory", default_model_dir)
rf_model_path = os.path.join(model_dir, "random_forest_model.joblib")
nb_model_path = os.path.join(model_dir, "naive_bayes_model.joblib")
gb_model_path = os.path.join(model_dir, "gradient_boosting_model.joblib")

validation_dir = st.sidebar.text_input("Validation Images Directory", default_validation_dir)
coco_json_path = os.path.join(validation_dir, "_annotations.coco.json")

# Check if model files exist
rf_exists = os.path.exists(rf_model_path)
nb_exists = os.path.exists(nb_model_path)
gb_exists = os.path.exists(gb_model_path)

if not rf_exists or not nb_exists:
    st.sidebar.error("❌ Models not found in the specified directory!")
    if not rf_exists:
        st.sidebar.write(f"Missing: `{os.path.basename(rf_model_path)}`")
    if not nb_exists:
        st.sidebar.write(f"Missing: `{os.path.basename(nb_model_path)}`")
    st.info("Please adjust the **Model Directory** in the sidebar to where `random_forest_model.joblib` and `naive_bayes_model.joblib` are saved.")
    st.stop()

if not gb_exists:
    st.sidebar.info("Optional: add `gradient_boosting_model.joblib` in the model directory to enable the Gradient Boosting predictor.")

# Load models
with st.spinner("Loading machine learning models..."):
    try:
        rf_model, nb_model, gb_model = load_models(
            rf_model_path,
            nb_model_path,
            gb_model_path if gb_exists else None,
        )
        st.sidebar.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# Load annotations
coco_data = load_coco_annotations(coco_json_path)

# Sidebar model selection
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Prediction Settings")
model_options = [
    "Ensemble (Average probability)",
    "Random Forest Classifier",
    "Naive Bayes Classifier",
]
if gb_exists:
    model_options.append("Gradient Boosting Classifier")

selected_model_name = st.sidebar.selectbox(
    "Primary Predictor",
    model_options
)

# ----------------- Tab Navigation -----------------

tab1, tab2, tab3 = st.tabs(["🎯 Single Scan Prediction", "📊 Batch Validation Metrics", "🔬 Model Metadata & Details"])

# --- TAB 1: Single Scan Prediction ---
with tab1:
    st.header("Single CT Scan Analysis")
    
    col_input, col_display = st.columns([1, 1])
    
    image_to_predict = None
    file_name_label = None
    ground_truth_label = None
    bboxes = []
    
    with col_input:
        st.subheader("Select Input CT Scan")
        input_source = st.radio("Choose scan source:", ["Validation Dataset Files", "Upload Custom Image"])
        
        if input_source == "Validation Dataset Files":
            if os.path.exists(validation_dir):
                all_files = [f for f in os.listdir(validation_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if all_files:
                    selected_file = st.selectbox("Select a validation scan:", all_files)
                    selected_path = os.path.join(validation_dir, selected_file)
                    
                    try:
                        image_to_predict = Image.open(selected_path)
                        file_name_label = selected_file
                        
                        # Determine Ground Truth from COCO if available
                        if coco_data:
                            # Map filename to image id
                            img_id = None
                            for img in coco_data.get("images", []):
                                if img.get("file_name") == selected_file:
                                    img_id = img.get("id")
                                    break
                            
                            if img_id is not None:
                                # Look up annotations
                                for ann in coco_data.get("annotations", []):
                                    if ann.get("image_id") == img_id:
                                        if ann.get("category_id") == 1:  # nodule
                                            ground_truth_label = 1  # Malignant
                                            bboxes.append(ann.get("bbox"))  # [x, y, w, h]
                        
                        if ground_truth_label is None:
                            # fallback: all files in COCO dataset are considered malignant (1)
                            ground_truth_label = 1
                            
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                else:
                    st.warning(f"No image files (.jpg, .jpeg, .png) found in `{validation_dir}`")
            else:
                st.warning(f"Validation directory not found at: `{validation_dir}`")
        
        else:
            uploaded_file = st.file_uploader("Upload CT Scan image...", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                try:
                    image_to_predict = Image.open(uploaded_file)
                    file_name_label = uploaded_file.name
                    ground_truth_label = None  # Unknown for custom uploaded image
                except Exception as e:
                    st.error(f"Error loading uploaded file: {e}")
        
        # Display Prediction triggers
        if image_to_predict is not None:
            st.markdown("---")
            run_btn = st.button("🚀 Analyze Scan", use_container_width=True)
    
    with col_display:
        st.subheader("Visualization & Prediction Result")
        
        if image_to_predict is not None:
            # Draw bbox if present
            display_img = image_to_predict.copy()
            if bboxes:
                draw = ImageDraw.Draw(display_img)
                for bbox in bboxes:
                    x, y, w, h = bbox
                    # COCO coordinates: [x, y, width, height]
                    draw.rectangle([x, y, x + w, y + h], outline="#00f2fe", width=3)
                st.caption("🟢 Green/Blue box shows the ground-truth nodule annotation from dataset.")
            
            st.image(display_img, caption=f"Scan: {file_name_label}", use_container_width=True)
            
            # Predict when button clicked or on load
            if 'run_btn' in locals() and run_btn:
                features = preprocess_image(image_to_predict)
                
                # Get probabilities
                rf_prob = rf_model.predict_proba(features)[0][1]
                nb_prob = nb_model.predict_proba(features)[0][1]
                gb_prob = gb_model.predict_proba(features)[0][1] if gb_model else None
                
                # Select probability based on settings
                if selected_model_name == "Random Forest Classifier":
                    prob = rf_prob
                elif selected_model_name == "Naive Bayes Classifier":
                    prob = nb_prob
                elif selected_model_name == "Gradient Boosting Classifier":
                    prob = gb_prob
                else:
                    probs = [rf_prob, nb_prob]
                    if gb_prob is not None:
                        probs.append(gb_prob)
                    prob = sum(probs) / len(probs)
                
                pred_label = 1 if prob >= 0.5 else 0
                pred_class = "Malignant (Nodule Present)" if pred_label == 1 else "Benign (No Nodule)"
                
                # Output Results Card
                st.markdown('<div class="card-container">', unsafe_allow_html=True)
                st.markdown(f"**Prediction Method:** {selected_model_name}")
                
                if pred_label == 1:
                    st.markdown(f"**Diagnosis:** <span class=\"status-badge-malignant\">{pred_class}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Diagnosis:** <span class=\"status-badge-benign\">{pred_class}</span>", unsafe_allow_html=True)
                
                st.markdown(f"**Confidence/Probability of Malignancy:** `{prob*100:.2f}%`")
                st.progress(float(prob))
                
                # Show Ground Truth Comparison
                if ground_truth_label is not None:
                    gt_class = "Malignant" if ground_truth_label == 1 else "Benign"
                    st.markdown(f"**Ground Truth Label:** `{gt_class}`")
                    if pred_label == ground_truth_label:
                        st.success("✅ Prediction MATCHES ground truth!")
                    else:
                        st.warning("⚠️ Prediction MISMATCHES ground truth.")
                
                st.markdown("</div>", unsafe_allow_html=True)

                # Recommendation guidance for clinician or patient next steps
                if pred_label == 1:
                    st.markdown(
                        """
                        <div class="card-container">
                            <strong>Recommended next steps</strong>
                            <ul>
                                <li>High malignancy likelihood detected. Recommend referral to a pulmonologist or oncologist.</li>
                                <li>Consider further diagnostic imaging (CT, PET) and clinical evaluation before treatment planning.</li>
                                <li>Advise follow-up testing rather than prescribing medication immediately.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="card-container">
                            <strong>Recommended next steps</strong>
                            <ul>
                                <li>Scan appears likely benign, but continue clinical monitoring as appropriate.</li>
                                <li>If symptoms persist, consult a physician for further evaluation.</li>
                                <li>Maintain preventive care such as smoking cessation and routine lung health checks.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.caption("⚠️ This tool is an aid only. Always consult a qualified healthcare professional for diagnosis and treatment.")

                render_patient_drug_guide(prob, pred_label)

                # Details columns for models
                if gb_model:
                    c1, c2, c3 = st.columns(3)
                else:
                    c1, c2 = st.columns(2)

                with c1:
                    st.metric("Random Forest Prob", f"{rf_prob*100:.1f}%")
                with c2:
                    st.metric("Naive Bayes Prob", f"{nb_prob*100:.1f}%")
                if gb_model:
                    with c3:
                        st.metric("Gradient Boosting Prob", f"{gb_prob*100:.1f}%")
        else:
            st.info("Select or upload a CT scan image to view predictions.")

# --- TAB 2: Batch Validation Metrics ---
with tab2:
    st.header("Validation Dataset Diagnostics Dashboard")
    st.write("Evaluate performance metrics on the validation dataset located on your machine.")
    st.info(f"Validation folder: `{validation_dir}`")
    if coco_data is None:
        st.warning("COCO annotation file `_annotations.coco.json` is not loaded. Ensure the selected validation folder contains this file.")
    
    if st.button("🔄 Run Validation Batch Evaluation", use_container_width=True):
        if not os.path.exists(validation_dir) or not coco_data:
            st.error("Validation directory or _annotations.coco.json not found. Check configuration in the sidebar.")
        else:
            # Find all images in validation folder
            images_list = coco_data.get("images", [])
            annotations_list = coco_data.get("annotations", [])
            
            # Map image IDs to filenames
            image_id_to_filename = {img["id"]: img["file_name"] for img in images_list}
            
            # Gather ground truths
            filename_to_gt = {img["file_name"]: 0 for img in images_list}
            for ann in annotations_list:
                img_id = ann.get("image_id")
                cat_id = ann.get("category_id")
                if cat_id == 1:  # nodule
                    fname = image_id_to_filename.get(img_id)
                    if fname:
                        filename_to_gt[fname] = 1  # Malignant
            
            # Filter filenames that actually exist
            valid_scans = []
            ground_truth = []
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            start_time = time.time()
            
            for idx, img_info in enumerate(images_list):
                fname = img_info["file_name"]
                fpath = os.path.join(validation_dir, fname)
                
                if os.path.exists(fpath):
                    valid_scans.append(fname)
                    ground_truth.append(filename_to_gt[fname])
                
                progress_bar.progress((idx + 1) / len(images_list))
                status_text.text(f"Scanning and loading image {idx+1}/{len(images_list)}...")
            
            if len(valid_scans) == 0:
                status_text.error("No validation images were found in the selected directory.")
                st.stop()
            
            status_text.text("Extracting image features and running predictions...")
            
            features_list = []
            for idx, fname in enumerate(valid_scans):
                fpath = os.path.join(validation_dir, fname)
                img = Image.open(fpath)
                feat = preprocess_image(img)
                features_list.append(feat[0])
            
            if not features_list:
                status_text.error("No valid image features could be extracted from the validation images.")
                st.stop()
            
            X_val = np.vstack(features_list)
            y_val = np.array(ground_truth)
            
            # Predict
            rf_preds = rf_model.predict(X_val)
            rf_probs = rf_model.predict_proba(X_val)[:, 1]
            
            nb_preds = nb_model.predict(X_val)
            nb_probs = nb_model.predict_proba(X_val)[:, 1]
            
            if gb_model:
                gb_preds = gb_model.predict(X_val)
                gb_probs = gb_model.predict_proba(X_val)[:, 1]
            else:
                gb_preds = None
                gb_probs = None
            
            # Ensemble predictions (avg probability > 0.5)
            prob_arrays = [rf_probs, nb_probs]
            if gb_probs is not None:
                prob_arrays.append(gb_probs)
            ensemble_probs = np.mean(np.vstack(prob_arrays), axis=0)
            ensemble_preds = (ensemble_probs >= 0.5).astype(int)
            
            duration = time.time() - start_time
            status_text.success(f"Inference complete on {len(valid_scans)} images in {duration:.2f} seconds.")
            
            # Show Metrics Card Columns
            if gb_model:
                mc1, mc2, mc3, mc4 = st.columns(4)
            else:
                mc1, mc2, mc3 = st.columns(3)

            with mc1:
                acc_rf = accuracy_score(y_val, rf_preds)
                st.markdown(f"""
                <div class="card-container">
                    <div class="metric-label">Random Forest Accuracy</div>
                    <div class="metric-value-large">{acc_rf*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with mc2:
                acc_nb = accuracy_score(y_val, nb_preds)
                st.markdown(f"""
                <div class="card-container">
                    <div class="metric-label">Naive Bayes Accuracy</div>
                    <div class="metric-value-large">{acc_nb*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            if gb_model:
                with mc3:
                    acc_gb = accuracy_score(y_val, gb_preds)
                    st.markdown(f"""
                    <div class="card-container">
                        <div class="metric-label">Gradient Boosting Accuracy</div>
                        <div class="metric-value-large">{acc_gb*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with mc4:
                    acc_ens = accuracy_score(y_val, ensemble_preds)
                    st.markdown(f"""
                    <div class="card-container">
                        <div class="metric-label">Ensemble Accuracy</div>
                        <div class="metric-value-large">{acc_ens*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with mc3:
                    acc_ens = accuracy_score(y_val, ensemble_preds)
                    st.markdown(f"""
                    <div class="card-container">
                        <div class="metric-label">Ensemble Accuracy</div>
                        <div class="metric-value-large">{acc_ens*100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed Plots
            st.subheader("Performance Visualizations")
            
            # Setup figures
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 1. Confusion Matrix (based on chosen predictor)
            if selected_model_name == "Random Forest Classifier":
                cm = confusion_matrix(y_val, rf_preds)
                title_cm = "Random Forest Confusion Matrix"
            elif selected_model_name == "Naive Bayes Classifier":
                cm = confusion_matrix(y_val, nb_preds)
                title_cm = "Naive Bayes Confusion Matrix"
            elif selected_model_name == "Gradient Boosting Classifier":
                cm = confusion_matrix(y_val, gb_preds)
                title_cm = "Gradient Boosting Confusion Matrix"
            else:
                cm = confusion_matrix(y_val, ensemble_preds)
                title_cm = "Ensemble Confusion Matrix"
            
            # If classes contain only 1 class in ground truth, confusion matrix might be 1x1
            # Adjust heatmap plot accordingly
            classes = ["Benign", "Malignant"]
            if cm.shape == (1, 1):
                # Ensure it's rendered correctly
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                            xticklabels=[classes[y_val[0]]], yticklabels=[classes[y_val[0]]])
            else:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                            xticklabels=classes, yticklabels=classes)
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('True')
            ax1.set_title(title_cm)
            
            # 2. ROC Curves
            roc_rf = safe_roc_curve(y_val, rf_probs)
            roc_nb = safe_roc_curve(y_val, nb_probs)
            roc_gb = safe_roc_curve(y_val, gb_probs) if gb_model is not None else None
            roc_ens = safe_roc_curve(y_val, ensemble_probs)

            roc_plotted = False
            if roc_rf is not None:
                fpr_rf, tpr_rf, _ = roc_rf
                auc_rf = auc(fpr_rf, tpr_rf)
                ax2.plot(fpr_rf, tpr_rf, color='blue', label=f'Random Forest (AUC = {auc_rf:.2f})')
                roc_plotted = True
            else:
                fpr_rf = tpr_rf = auc_rf = None

            if roc_nb is not None:
                fpr_nb, tpr_nb, _ = roc_nb
                auc_nb = auc(fpr_nb, tpr_nb)
                ax2.plot(fpr_nb, tpr_nb, color='purple', label=f'Naive Bayes (AUC = {auc_nb:.2f})')
                roc_plotted = True
            else:
                fpr_nb = tpr_nb = auc_nb = None

            if roc_gb is not None:
                fpr_gb, tpr_gb, _ = roc_gb
                auc_gb = auc(fpr_gb, tpr_gb)
                ax2.plot(fpr_gb, tpr_gb, color='orange', label=f'Gradient Boosting (AUC = {auc_gb:.2f})')
                roc_plotted = True
            else:
                fpr_gb = tpr_gb = auc_gb = None

            if roc_ens is not None:
                fpr_ens, tpr_ens, _ = roc_ens
                auc_ens = auc(fpr_ens, tpr_ens)
                ax2.plot(fpr_ens, tpr_ens, color='green', label=f'Ensemble (AUC = {auc_ens:.2f})')
                roc_plotted = True
            else:
                fpr_ens = tpr_ens = auc_ens = None

            ax2.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random guess')

            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curves comparison')
            if roc_plotted:
                ax2.legend(loc='lower right')
            else:
                ax2.text(0.5, 0.5, 'ROC not available\n(single-class labels only)', ha='center', va='center', fontsize=12)
            ax2.grid(True)
            
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("Classification Report")
            if selected_model_name == "Random Forest Classifier":
                report_dict = classification_report(y_val, rf_preds, output_dict=True, zero_division=0)
            elif selected_model_name == "Naive Bayes Classifier":
                report_dict = classification_report(y_val, nb_preds, output_dict=True, zero_division=0)
            elif selected_model_name == "Gradient Boosting Classifier":
                report_dict = classification_report(y_val, gb_preds, output_dict=True, zero_division=0)
            else:
                report_dict = classification_report(y_val, ensemble_preds, output_dict=True, zero_division=0)
            
            st.json(report_dict)

# --- TAB 3: Model Metadata & Details ---
with tab3:
    st.header("Lung Nodule Classification System Models")
    
    st.markdown("""
    This prediction system consists of two shallow machine learning models trained on a merged dataset composed of **LUNA16** (pre-processed lung slices) and **COCO Lung Dataset** CT scans.
    
    ### Pipeline Architecture
    1. **Preprocessing Layer**:
       - Re-scaling pixel values to `[0.0, 1.0]` range.
       - Resizing spatial dimensions to `224 x 224` pixels.
       - Grayscale to 3-channel (RGB) expansion.
       - Flattening 3D input tensor `(224, 224, 3)` to a 1D vector of `150,528` dimensions.
    2. **Classifier Models**:
       - **Random Forest Classifier**: Formed of 100 decision estimators, configured with balanced class weights to address training dataset imbalance.
       - **Naive Bayes Classifier**: Gaussian Naive Bayes model modeling high-dimensional pixel density.
    """)
    
    st.subheader("Loaded Model Properties")
    
    if gb_model:
        prop_col1, prop_col2, prop_col3 = st.columns(3)
    else:
        prop_col1, prop_col2 = st.columns(2)

    with prop_col1:
        st.markdown("**Random Forest Classifier Details**")
        st.markdown(f"- **Expected Input Features:** `{rf_model.n_features_in_}`")
        st.markdown(f"- **Number of estimators:** `{len(rf_model.estimators_)}`")
        st.markdown(f"- **Classes learned:** `{rf_model.classes_}`")
    with prop_col2:
        st.markdown("**Naive Bayes Classifier Details**")
        st.markdown(f"- **Expected Input Features:** `{nb_model.n_features_in_}`")
        st.markdown(f"- **Classes learned:** `{nb_model.classes_}`")
    if gb_model:
        with prop_col3:
            st.markdown("**Gradient Boosting Classifier Details**")
            st.markdown(f"- **Expected Input Features:** `{gb_model.n_features_in_}`")
            st.markdown(f"- **Number of estimators:** `{len(gb_model.estimators_)}`")
            st.markdown(f"- **Classes learned:** `{gb_model.classes_}`")
