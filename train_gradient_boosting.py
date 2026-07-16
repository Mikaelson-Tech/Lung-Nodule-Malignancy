import os
import json
import argparse
from PIL import Image
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_coco_labels(coco_json_path, images_dir):
    labels = {}
    # Load COCO JSON labels if present
    if os.path.exists(coco_json_path):
        with open(coco_json_path, 'r') as f:
            coco = json.load(f)
        img_id_to_name = {img['id']: img['file_name'] for img in coco.get('images', [])}
        anns = coco.get('annotations', [])

        # Default label 0 for all images listed in COCO, set to 1 if any annotation with category_id==1
        labels = {name: 0 for name in img_id_to_name.values()}
        for ann in anns:
            img_name = img_id_to_name.get(ann.get('image_id'))
            if img_name and ann.get('category_id') == 1:
                labels[img_name] = 1

    # Collect all image files in images_dir (include jpg/jpeg/png)
    items = []
    if os.path.exists(images_dir):
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            fpath = os.path.join(images_dir, fname)
            # If file is listed in COCO labels use that label, otherwise treat as benign (0)
            lbl = labels.get(fname, 0)
            if os.path.exists(fpath):
                items.append((fpath, lbl))
    return items


def preprocess_image(path, size=(224, 224)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten()


def main(args):
    train_images_dir = os.path.join(args.data_dir, 'train')
    valid_images_dir = os.path.join(args.data_dir, 'valid')
    train_json = os.path.join(train_images_dir, '_annotations.coco.json')
    valid_json = os.path.join(valid_images_dir, '_annotations.coco.json')

    if not os.path.exists(train_json):
        raise SystemExit(f"Train COCO annotations not found at: {train_json}")

    print('Loading training labels...')
    train_items = load_coco_labels(train_json, train_images_dir)
    print(f'Found {len(train_items)} training images')

    if len(train_items) == 0:
        raise SystemExit('No training images found. Check dataset path')

    X = np.vstack([preprocess_image(p) for p, _ in train_items])
    y = np.array([lbl for _, lbl in train_items])

    unique = np.unique(y)
    print('Training label distribution:', {int(k): int((y==k).sum()) for k in unique})
    if unique.size < 2:
        raise SystemExit('Only one class present in training labels; cannot train classifier.')

    print('Training GradientBoostingClassifier...')
    clf = GradientBoostingClassifier(n_estimators=args.n_estimators)
    clf.fit(X, y)

    # Evaluate on validation set if available
    if os.path.exists(valid_json):
        print('Loading validation set...')
        valid_items = load_coco_labels(valid_json, valid_images_dir)
        if valid_items:
            Xv = np.vstack([preprocess_image(p) for p, _ in valid_items])
            yv = np.array([lbl for _, lbl in valid_items])
            preds = clf.predict(Xv)
            probs = clf.predict_proba(Xv)[:, 1] if hasattr(clf, 'predict_proba') else None
            print('Validation accuracy:', accuracy_score(yv, preds))
            print('Classification report:')
            print(classification_report(yv, preds, zero_division=0))
        else:
            print('No validation images found; skipping evaluation')
    else:
        print('Validation annotations not found; skipping evaluation')

    print(f'Saving model to {args.output}')
    joblib.dump(clf, args.output)
    print('Done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train GradientBoostingClassifier on COCO lung dataset and export .joblib')
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), 'lung_ct_version_n_512.v2i.coco'), help='Path to COCO dataset folder')
    p.add_argument('--output', default=os.path.join(os.path.dirname(__file__), 'gradient_boosting_model.joblib'), help='Output model path')
    p.add_argument('--n-estimators', type=int, default=100, help='Number of estimators for GradientBoosting')
    args = p.parse_args()
    main(args)
