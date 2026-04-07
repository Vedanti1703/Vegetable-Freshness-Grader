from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

app = FastAPI()

# Enable CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 LOAD CHECKPOINT (IMPORTANT FIX)
checkpoint = torch.load("freshness_model.pth", map_location=device)

# 🔥 RECREATE MODEL (same as training)
model = models.resnet18(pretrained=False)

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Get class names dynamically
classes = checkpoint['class_names']

# Image preprocessing (MATCH TRAINING)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "API running"}

# 🔥 MATCHES YOUR FRONTEND (/analyze)
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        image = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        prediction = classes[predicted.item()]
        confidence = confidence.item() * 100

        # 🔥 Convert to frontend format
        if prediction.lower() == "fresh":
            grade = "A"
            freshness_score = int(confidence)
            shelf_life = 5
            recommendation = "Safe to consume"
            rot = 5
            wilting = 10
            edge = 8
            saturation = 180
            issues = []
        else:
            grade = "D"
            freshness_score = int(confidence)
            shelf_life = 1
            recommendation = "Avoid consuming"
            rot = 60
            wilting = 70
            edge = 25
            saturation = 80
            issues = ["Rot detected", "Wilting"]

        return {
            "status": "success",
            "data": {
                "grade": grade,
                "vegetable_name": "Vegetable",
                "hindi_name": "सब्जी",
                "freshness_score": freshness_score,
                "recommendation": recommendation,
                "shelf_life_days": shelf_life,
                "price_min_inr_per_kg": 20,
                "price_max_inr_per_kg": 60,

                "cv_metrics": {
                    "saturation": saturation,
                    "rot_pct": rot,
                    "wilting_pct": wilting,
                    "edge_density": edge
                },

                "yolo_hint": {
                    "name": "Vegetable",
                    "confidence": round(confidence/100, 2)
                },

                "ml_prediction": prediction,
                "ml_confidence": round(confidence, 2),

                "issues": issues
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }