import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image

# ══════════════════════════════════════════════════════════════════
# LOAD MODELS AT STARTUP
# ══════════════════════════════════════════════════════════════════
yolo = YOLO("yolov8n.pt")

# Load custom freshness classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
freshness_model = models.resnet18(pretrained=False)
freshness_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(freshness_model.fc.in_features, 2)
)

# Load trained weights
checkpoint = torch.load("freshness_model.pth", map_location=device)
freshness_model.load_state_dict(checkpoint['model_state_dict'])
freshness_model = freshness_model.to(device)
freshness_model.eval()

print(f"✅ Models loaded successfully on {device}")

# ══════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ══════════════════════════════════════════════════════════════════
# VEGETABLE DATABASE
# ══════════════════════════════════════════════════════════════════

# YOLO class to vegetable name mapping
VEGETABLE_NAMES = {
    "apple": "Apple", "banana": "Banana", "orange": "Orange",
    "carrot": "Carrot", "broccoli": "Broccoli", "potato": "Potato",
    "tomato": "Tomato", "onion": "Onion", "cucumber": "Cucumber",
    "hot dog": "Sausage", "donut": "Potato",  # YOLO often confuses these
    "cake": "Cabbage", "pizza": "Tomato"
}

# Hindi name mapping
HINDI_NAMES = {
    "Potato": "आलू (Aloo)", 
    "Tomato": "टमाटर (Tamatar)", 
    "Onion": "प्याज (Pyaaz)",
    "Carrot": "गाजर (Gajar)", 
    "Cabbage": "पत्तागोभी (Patta Gobi)", 
    "Bell Pepper": "शिमला मिर्च (Shimla Mirch)",
    "Broccoli": "हरी गोभी (Hari Gobi)", 
    "Cucumber": "खीरा (Kheera)", 
    "Apple": "सेब (Seb)",
    "Banana": "केला (Kela)", 
    "Orange": "संतरा (Santra)", 
    "Lettuce": "सलाद पत्ता (Salad Patta)",
    "Sausage": "सॉसेज (Sausage)"
}

# Price ranges for Indian markets (INR/kg)
PRICE_RANGES = {
    "Potato": (20, 40), "Tomato": (20, 60), "Onion": (15, 35),
    "Carrot": (30, 50), "Cabbage": (15, 30), "Bell Pepper": (40, 80),
    "Broccoli": (60, 100), "Cucumber": (20, 40), "Apple": (80, 150),
    "Banana": (30, 60), "Orange": (40, 80), "Lettuce": (40, 70),
    "Sausage": (200, 400)
}

# ════════════════════════════════════════
# STAGE 1 — OpenCV Feature Extraction
# ════════════════════════════════════════
def extract_cv_features(img_array):
    """
    Extract computer vision features from the image:
    - Color saturation and brightness
    - Wilting percentage
    - Rot/dark spot percentage
    - Edge density (wrinkle detection)
    """
    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

    saturation   = float(hsv[:,:,1].mean())
    brightness   = float(hsv[:,:,2].mean())

    # Detect wilting (low saturation areas)
    wilting_mask = hsv[:,:,1] < 40
    wilting_pct  = round(float(wilting_mask.mean()) * 100, 2)

    # Detect rot (very dark areas)
    lower_dark   = np.array([0,   0,   0])
    upper_dark   = np.array([180, 255, 60])
    dark_mask    = cv2.inRange(hsv, lower_dark, upper_dark)
    rot_pct      = round((dark_mask.sum()/255) / dark_mask.size * 100, 2)

    # Detect wrinkles (edge density)
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

# ════════════════════════════════════════
# STAGE 2 — YOLO Object Detection
# ════════════════════════════════════════
def detect_vegetable_yolo(img_array):
    """
    Use YOLO to detect and identify the vegetable in the image.
    Returns the highest confidence detection.
    """
    results    = yolo.predict(source=img_array, conf=0.25, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            class_name = yolo.names[int(box.cls)]
            mapped_name = VEGETABLE_NAMES.get(class_name, class_name.title())
            
            detections.append({
                "name"      : mapped_name,
                "confidence": round(float(box.conf), 2)
            })

    if not detections:
        return {"name": "Vegetable", "confidence": 0.0}

    return max(detections, key=lambda x: x["confidence"])

# ════════════════════════════════════════
# STAGE 3 — CNN Freshness Classification
# ════════════════════════════════════════
def classify_freshness(img_array):
    """
    Use custom trained ResNet18 model to classify fresh vs rotten.
    Returns (is_fresh: bool, confidence: float)
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Preprocess and predict
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = freshness_model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Assuming class 0 = fresh, class 1 = rotten
    is_fresh = predicted.item() == 0
    confidence_score = confidence.item() * 100
    
    return is_fresh, confidence_score

# ════════════════════════════════════════
# STAGE 4 — Grade Calculation
# ════════════════════════════════════════
def calculate_grade(cv_features, is_fresh, freshness_confidence):
    """
    Calculate grade (A/B/C/D) based on:
    1. CV metrics (rot, wilting, saturation)
    2. ML model prediction
    3. Confidence level
    """
    rot_pct = cv_features['rot_pct']
    wilting_pct = cv_features['wilting_pct']
    saturation = cv_features['saturation']
    
    # If model strongly predicts rotten, downgrade
    if not is_fresh and freshness_confidence > 70:
        if rot_pct > 30 or wilting_pct > 40:
            return "D"
        else:
            return "C"
    
    # If model predicts rotten with medium confidence
    if not is_fresh and freshness_confidence > 50:
        if rot_pct < 15 and wilting_pct < 25:
            return "B"  # Maybe just aging
        else:
            return "C"
    
    # Grade based primarily on CV metrics
    if rot_pct < 5 and wilting_pct < 10 and saturation > 150:
        return "A"  # Very fresh
    elif rot_pct < 15 and wilting_pct < 25 and saturation > 100:
        return "B"  # Fresh
    elif rot_pct < 30 and wilting_pct < 40 and saturation > 70:
        return "C"  # Aging
    else:
        return "D"  # Avoid

# ════════════════════════════════════════
# STAGE 5 — Result Generation
# ════════════════════════════════════════
def generate_result(vegetable_name, grade, cv_features, is_fresh, freshness_confidence):
    """
    Generate the complete analysis result with all details.
    """
    
    # Calculate overall freshness score (0-100)
    base_score = 100
    base_score -= cv_features['rot_pct'] * 1.5
    base_score -= cv_features['wilting_pct'] * 1.0
    base_score += (cv_features['saturation'] / 255) * 20
    
    # Apply ML model influence
    if not is_fresh:
        base_score *= (1 - (freshness_confidence / 200))  # Reduce based on confidence
    
    freshness_score = max(0, min(100, int(base_score)))
    
    # Estimate shelf life based on grade
    shelf_life_map = {"A": 7, "B": 5, "C": 2, "D": 1}
    shelf_life = shelf_life_map.get(grade, 1)
    
    # Calculate price range with grade-based discount
    base_min, base_max = PRICE_RANGES.get(vegetable_name, (20, 50))
    price_multipliers = {
        "A": 1.0,   # Full price
        "B": 0.9,   # 10% off
        "C": 0.6,   # 40% off
        "D": 0.3    # 70% off (or don't buy)
    }
    multiplier = price_multipliers.get(grade, 0.5)
    
    price_min = int(base_min * multiplier)
    price_max = int(base_max * multiplier)
    
    # Detect specific issues
    issues = []
    if cv_features['rot_pct'] > 15:
        issues.append("Dark spots detected")
    if cv_features['wilting_pct'] > 25:
        issues.append("Wilting visible")
    if cv_features['saturation'] < 100:
        issues.append("Low color saturation")
    if cv_features['edge_density'] > 15:
        issues.append("Surface wrinkles present")
    if not is_fresh and freshness_confidence > 60:
        issues.append("ML model detected spoilage")
    
    # Generate recommendation
    recommendations = {
        "A": "Perfect for storage and resale • Buy at full price",
        "B": "Good buy • Consume within a week",
        "C": "Use immediately • Negotiate 30-40% discount",
        "D": "Avoid purchase • Food safety risk"
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

# ════════════════════════════════════════
# FULL PIPELINE
# ════════════════════════════════════════
def run_pipeline(img_array):
    """
    Complete analysis pipeline:
    1. Extract OpenCV features (color, rot, wilting, edges)
    2. Detect vegetable type with YOLO
    3. Classify freshness with custom CNN model
    4. Calculate grade using hybrid approach
    5. Generate complete result with pricing and recommendations
    """
    
    # Stage 1 — OpenCV feature extraction
    cv_features = extract_cv_features(img_array)
    
    # Stage 2 — YOLO vegetable detection
    yolo_hint = detect_vegetable_yolo(img_array)
    
    # Stage 3 — CNN freshness classification
    is_fresh, freshness_confidence = classify_freshness(img_array)
    
    # Stage 4 — Calculate grade
    grade = calculate_grade(cv_features, is_fresh, freshness_confidence)
    
    # Stage 5 — Generate complete result
    result = generate_result(
        yolo_hint['name'], 
        grade, 
        cv_features, 
        is_fresh, 
        freshness_confidence
    )
    
    # Attach raw data for frontend display
    result["cv_metrics"] = cv_features
    result["yolo_hint"] = yolo_hint
    
    return result
