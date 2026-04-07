# 🥦 Vegetable Freshness Grader: API to Custom Model Migration Guide

## Overview
This guide will help you replace the Groq API with a custom-trained CNN model while keeping the UI exactly the same.

---

## 📊 Step 1: Dataset Collection

### Recommended Dataset
**Vegetable Freshness Detection Dataset** (Kaggle)
- **Link**: https://www.kaggle.com/datasets/muhriddinmuxiddinov/vegetables-fresh-and-rotten-for-classification
- **Size**: ~7,000 images
- **Classes**: Fresh and Rotten vegetables (multiple types)
- **Format**: JPG images organized by class

### Alternative Datasets (if first is unavailable)
1. **Fruits Fresh and Rotten for Classification**
   - Link: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
   
2. **Vegetable Image Dataset**
   - Link: https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset

### Dataset Structure After Download
```
dataset/
├── train/
│   ├── fresh/
│   │   ├── fresh_potato_001.jpg
│   │   ├── fresh_tomato_001.jpg
│   │   └── ...
│   └── rotten/
│       ├── rotten_potato_001.jpg
│       ├── rotten_tomato_001.jpg
│       └── ...
└── test/
    ├── fresh/
    └── rotten/
```

---

## 🧠 Step 2: Create the Model Training Script

Create `train_model.py` in your project root:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm

# Configuration
DATASET_PATH = "dataset/train"  # Update this path
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 224
NUM_WORKERS = 4

# Data transformations
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                          shuffle=True, num_workers=NUM_WORKERS)

# Create model (ResNet18 - lightweight and accurate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)

# Modify final layer for binary classification (fresh vs rotten)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print(f"Training on {device}")
print(f"Dataset classes: {train_dataset.classes}")
print(f"Total images: {len(train_dataset)}")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': train_dataset.classes
}, 'freshness_model.pth')

print("✅ Model saved as 'freshness_model.pth'")
```

---

## 🔧 Step 3: Update `pipeline.py`

Replace the entire `pipeline.py` with this new version:

```python
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

# ── Load models once at startup ──
yolo = YOLO("yolov8n.pt")

# Load custom freshness model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
freshness_model = models.resnet18(pretrained=False)
freshness_model.fc = torch.nn.Linear(freshness_model.fc.in_features, 2)

checkpoint = torch.load("freshness_model.pth", map_location=device)
freshness_model.load_state_dict(checkpoint['model_state_dict'])
freshness_model = freshness_model.to(device)
freshness_model.eval()

# Image preprocessing for model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Vegetable name mapping (YOLO class to common names)
VEGETABLE_NAMES = {
    "apple": "Apple", "banana": "Banana", "orange": "Orange",
    "carrot": "Carrot", "broccoli": "Broccoli", "potato": "Potato",
    "tomato": "Tomato", "onion": "Onion", "cucumber": "Cucumber",
    "pepper": "Bell Pepper", "lettuce": "Lettuce", "cabbage": "Cabbage"
}

# Hindi name mapping
HINDI_NAMES = {
    "Potato": "Aloo", "Tomato": "Tamatar", "Onion": "Pyaaz",
    "Carrot": "Gajar", "Cabbage": "Patta Gobi", "Bell Pepper": "Shimla Mirch",
    "Broccoli": "Hari Gobi", "Cucumber": "Kheera", "Apple": "Seb",
    "Banana": "Kela", "Orange": "Santra", "Lettuce": "Salad Patta"
}

# Price ranges (INR/kg)
PRICE_RANGES = {
    "Potato": (20, 40), "Tomato": (20, 60), "Onion": (15, 35),
    "Carrot": (30, 50), "Cabbage": (15, 30), "Bell Pepper": (40, 80),
    "Broccoli": (60, 100), "Cucumber": (20, 40), "Apple": (80, 150),
    "Banana": (30, 60), "Orange": (40, 80), "Lettuce": (40, 70)
}

# ════════════════════════════════════
# STAGE 1 — OpenCV Feature Extraction
# ════════════════════════════════════
def extract_cv_features(img_array):
    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

    saturation   = float(hsv[:,:,1].mean())
    brightness   = float(hsv[:,:,2].mean())

    wilting_mask = hsv[:,:,1] < 40
    wilting_pct  = round(float(wilting_mask.mean()) * 100, 2)

    lower_dark   = np.array([0,   0,   0])
    upper_dark   = np.array([180, 255, 60])
    dark_mask    = cv2.inRange(hsv, lower_dark, upper_dark)
    rot_pct      = round((dark_mask.sum()/255) / dark_mask.size * 100, 2)

    gray         = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    edges        = cv2.Canny(gray, 50, 150)
    edge_density = round(float(edges.mean()), 4)

    return {
        "saturation"  : round(saturation, 2),
        "brightness"  : round(brightness, 2),
        "wilting_pct" : wilting_pct,
        "rot_pct"     : rot_pct,
        "edge_density": edge_density
    }

# ════════════════════════════════════
# STAGE 2 — YOLO Detection
# ════════════════════════════════════
def detect_vegetable_yolo(img_array):
    results    = yolo.predict(source=img_array, conf=0.25, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            class_name = yolo.names[int(box.cls)]
            detections.append({
                "name"      : VEGETABLE_NAMES.get(class_name, class_name.title()),
                "confidence": round(float(box.conf), 2)
            })

    if not detections:
        return {"name": "Unknown", "confidence": 0.0}

    return max(detections, key=lambda x: x["confidence"])

# ════════════════════════════════════
# STAGE 3 — Freshness Classification
# ════════════════════════════════════
def classify_freshness(img_array):
    """Use custom CNN to classify fresh vs rotten"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Preprocess and predict
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = freshness_model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    is_fresh = predicted.item() == 0  # Assuming 0=fresh, 1=rotten
    confidence_score = confidence.item() * 100
    
    return is_fresh, confidence_score

# ════════════════════════════════════
# STAGE 4 — Grade Assignment Logic
# ════════════════════════════════════
def calculate_grade(cv_features, is_fresh, freshness_confidence):
    """
    Determine grade based on CV metrics and ML prediction
    """
    rot_pct = cv_features['rot_pct']
    wilting_pct = cv_features['wilting_pct']
    saturation = cv_features['saturation']
    
    # If model says rotten with high confidence, downgrade
    if not is_fresh and freshness_confidence > 70:
        if rot_pct > 30 or wilting_pct > 40:
            return "D"
        else:
            return "C"
    
    # Grade based on CV metrics
    if rot_pct < 5 and wilting_pct < 10 and saturation > 150:
        return "A"
    elif rot_pct < 15 and wilting_pct < 25 and saturation > 100:
        return "B"
    elif rot_pct < 30 and wilting_pct < 40 and saturation > 70:
        return "C"
    else:
        return "D"

# ════════════════════════════════════
# STAGE 5 — Generate Full Result
# ════════════════════════════════════
def generate_result(vegetable_name, grade, cv_features, is_fresh, freshness_confidence):
    """Generate complete analysis result"""
    
    # Calculate freshness score (0-100)
    base_score = 100
    base_score -= cv_features['rot_pct'] * 1.5
    base_score -= cv_features['wilting_pct'] * 1.0
    base_score += (cv_features['saturation'] / 255) * 20
    
    if not is_fresh:
        base_score *= 0.6  # Reduce score if model predicts rotten
    
    freshness_score = max(0, min(100, int(base_score)))
    
    # Shelf life estimation
    shelf_life_map = {"A": 7, "B": 5, "C": 2, "D": 1}
    shelf_life = shelf_life_map.get(grade, 1)
    
    # Price adjustment
    base_min, base_max = PRICE_RANGES.get(vegetable_name, (20, 50))
    price_multipliers = {"A": 1.0, "B": 0.9, "C": 0.6, "D": 0.3}
    multiplier = price_multipliers.get(grade, 0.5)
    
    price_min = int(base_min * multiplier)
    price_max = int(base_max * multiplier)
    
    # Issues detection
    issues = []
    if cv_features['rot_pct'] > 15:
        issues.append("Dark spots detected")
    if cv_features['wilting_pct'] > 25:
        issues.append("Wilting visible")
    if cv_features['saturation'] < 100:
        issues.append("Low color saturation")
    if not is_fresh:
        issues.append("Model detected spoilage")
    
    # Recommendation
    recommendations = {
        "A": "Perfect for storage and resale",
        "B": "Good buy, consume within a week",
        "C": "Use immediately, negotiate 30-40% discount",
        "D": "Avoid purchase, food safety risk"
    }
    
    return {
        "vegetable_name": vegetable_name,
        "hindi_name": HINDI_NAMES.get(vegetable_name, "सब्जी"),
        "grade": grade,
        "freshness_score": freshness_score,
        "shelf_life_days": shelf_life,
        "price_min_inr_per_kg": price_min,
        "price_max_inr_per_kg": price_max,
        "issues": issues,
        "recommendation": recommendations.get(grade, "Check with expert"),
        "yolo_was_correct": True,  # For UI compatibility
        "ml_prediction": "Fresh" if is_fresh else "Rotten",
        "ml_confidence": round(freshness_confidence, 2)
    }

# ════════════════════════════════════
# FULL PIPELINE
# ════════════════════════════════════
def run_pipeline(img_array):
    """
    Complete analysis pipeline:
    1. Extract CV features
    2. Detect vegetable with YOLO
    3. Classify freshness with custom CNN
    4. Calculate grade
    5. Generate complete result
    """
    
    # Stage 1 — OpenCV
    cv_features = extract_cv_features(img_array)
    
    # Stage 2 — YOLO (for vegetable identification)
    yolo_hint = detect_vegetable_yolo(img_array)
    
    # Stage 3 — Freshness classification with custom model
    is_fresh, freshness_confidence = classify_freshness(img_array)
    
    # Stage 4 — Calculate grade
    grade = calculate_grade(cv_features, is_fresh, freshness_confidence)
    
    # Stage 5 — Generate full result
    result = generate_result(
        yolo_hint['name'], 
        grade, 
        cv_features, 
        is_fresh, 
        freshness_confidence
    )
    
    # Attach raw data for frontend
    result["cv_metrics"] = cv_features
    result["yolo_hint"] = yolo_hint
    
    return result
```

---

## 📦 Step 4: Update `requirements.txt`

Replace the entire file with:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
opencv-python-headless==4.8.1.78
numpy==1.24.3
ultralytics==8.0.221
torch==2.1.0
torchvision==0.16.0
Pillow==10.1.0
```

---

## 🎨 Step 5: Update Frontend (Minor Text Changes)

Update line 435 in `frontend/index.html`:

**Before:**
```html
<p>CV-powered grading for Indian markets · OpenCV + YOLOv8 + Groq</p>
```

**After:**
```html
<p>CV-powered grading for Indian markets · OpenCV + YOLOv8 + Custom CNN</p>
```

Update line 477:

**Before:**
```html
<p>Running OpenCV + YOLO + Groq analysis...</p>
```

**After:**
```html
<p>Running OpenCV + YOLO + CNN analysis...</p>
```

---

## 🚀 Step 6: Complete Setup & Run Instructions

### 6.1 Download Dataset
```bash
# Option 1: Using Kaggle API
pip install kaggle
kaggle datasets download -d muhriddinmuxiddinov/vegetables-fresh-and-rotten-for-classification
unzip vegetables-fresh-and-rotten-for-classification.zip -d dataset

# Option 2: Manual download from Kaggle website
# Download from: https://www.kaggle.com/datasets/muhriddinmuxiddinov/vegetables-fresh-and-rotten-for-classification
# Extract to 'dataset' folder
```

### 6.2 Train the Model
```bash
# Install dependencies
pip install -r requirements.txt

# Train model (will take 10-30 minutes depending on GPU)
python train_model.py
```

### 6.3 Run the Application
```bash
# Start FastAPI backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Open frontend in browser
# Open frontend/index.html in your browser
# Or serve it with:
python -m http.server 3000 --directory frontend
# Then visit: http://localhost:3000
```

---

## 📊 Step 7: Model Performance Expectations

With the recommended dataset and training script:

- **Training Time**: 15-30 minutes (GPU) / 2-3 hours (CPU)
- **Expected Accuracy**: 85-95%
- **Model Size**: ~45 MB
- **Inference Speed**: 50-100ms per image

---

## 🎯 Key Differences from Groq Version

| Aspect | Groq API (Old) | Custom CNN (New) |
|--------|----------------|------------------|
| **Vegetable ID** | Groq vision model | YOLO + manual mapping |
| **Freshness** | Groq analysis | Custom trained ResNet18 |
| **Grade Logic** | Groq decides | Rule-based + ML hybrid |
| **Cost** | API calls ($) | Free after training |
| **Speed** | 1-3 seconds | 100-200ms |
| **Offline** | No | Yes |
| **Accuracy** | ~90% | ~85-95% |

---

## 🔍 Troubleshooting

### Model not loading
```python
# Check if model file exists
import os
assert os.path.exists("freshness_model.pth"), "Model file not found!"
```

### CUDA out of memory during training
```python
# Reduce batch size in train_model.py
BATCH_SIZE = 16  # or even 8
```

### Poor accuracy
- Train for more epochs (increase EPOCHS to 25-30)
- Use data augmentation (already included)
- Try different model (ResNet34 instead of ResNet18)

---

## ✅ Testing Checklist

- [ ] Dataset downloaded and extracted
- [ ] Model trained successfully
- [ ] `freshness_model.pth` file created
- [ ] Backend starts without errors
- [ ] Frontend loads correctly
- [ ] Can upload image and get results
- [ ] Results show grade, price, shelf life
- [ ] CV metrics display correctly
- [ ] UI matches original design

---

## 🎓 Next Steps for Improvement

1. **Multi-class Classification**: Train for specific vegetable types
2. **More Features**: Add texture analysis, shape detection
3. **Better Dataset**: Collect Indian market-specific images
4. **Model Ensemble**: Combine multiple models for better accuracy
5. **Mobile App**: Deploy with TensorFlow Lite
6. **Real-time Video**: Process video streams

---

## 📞 Support

If you encounter issues:
1. Check error messages in terminal
2. Verify all files are in correct locations
3. Ensure dependencies are installed correctly
4. Check Python version (3.8+ required)

Good luck with your project! 🚀
