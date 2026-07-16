import os
import joblib
import numpy as np
from PIL import Image

def verify():
    # 1. Paths
    model_dir = os.path.dirname(os.path.abspath(__file__))
    rf_model_path = os.path.join(model_dir, "random_forest_model.joblib")
    nb_model_path = os.path.join(model_dir, "naive_bayes_model.joblib")
    
    validation_dir = r"c:\Users\Ayoola\Downloads\Lung_dataset_2\lung_ct_version_n_512.v2i.coco\valid"
    
    # 2. Check model exists
    print("Checking model files:")
    print("RF exists:", os.path.exists(rf_model_path))
    print("NB exists:", os.path.exists(nb_model_path))
    
    # 3. Load models
    print("\nLoading models...")
    rf_model = joblib.load(rf_model_path)
    nb_model = joblib.load(nb_model_path)
    print("Models loaded successfully.")
    
    # 4. Check validation images
    if not os.path.exists(validation_dir):
        print(f"\n[ERROR] Validation directory not found at: {validation_dir}")
        return
        
    all_files = [f for f in os.listdir(validation_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nFound {len(all_files)} validation images.")
    if not all_files:
        print("[ERROR] No images found in validation directory.")
        return
        
    sample_file = all_files[0]
    sample_path = os.path.join(validation_dir, sample_file)
    print(f"Testing with sample image: {sample_file}")
    
    # 5. Preprocess
    img = Image.open(sample_path)
    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize((224, 224))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_flat = img_arr.flatten().reshape(1, -1)
    
    print("\nPreprocessed shape:", img_flat.shape)
    print("Expected shape:", rf_model.n_features_in_)
    
    if img_flat.shape[1] != rf_model.n_features_in_:
        print("[ERROR] Feature dimension mismatch!")
        return
        
    # 6. Predict
    print("\nRunning RF prediction...")
    rf_prob = rf_model.predict_proba(img_flat)[0][1]
    rf_pred = rf_model.predict(img_flat)[0]
    print(f"RF prediction: {rf_pred} (prob: {rf_prob:.4f})")
    
    print("Running NB prediction...")
    nb_prob = nb_model.predict_proba(img_flat)[0][1]
    nb_pred = nb_model.predict(img_flat)[0]
    print(f"NB prediction: {nb_pred} (prob: {nb_prob:.4f})")
    
    print("\n[OK] Verification script completed successfully. Pipeline is fully functional!")

if __name__ == "__main__":
    verify()
