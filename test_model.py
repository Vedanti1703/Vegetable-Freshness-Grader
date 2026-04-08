"""Quick test to debug model loading."""
import traceback
import os

print("=" * 50)
print("Testing model loading...")
print("=" * 50)

path = os.path.join(os.getcwd(), "freshness_model.pth")
print(f"Model path: {path}")
print(f"File exists: {os.path.exists(path)}")
print(f"File size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")

try:
    import torch
    import torch.nn as nn
    print(f"PyTorch version: {torch.__version__}")
    
    import torchvision
    print(f"Torchvision version: {torchvision.__version__}")
    
    from torchvision import models
    
    device = torch.device("cpu")
    
    # Step 1: Load checkpoint
    print("\n[1] Loading checkpoint...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    print(f"    Keys: {list(checkpoint.keys())}")
    print(f"    Classes: {checkpoint.get('class_names')}")
    
    # Step 2: Create model with weights=None (new API)
    print("\n[2] Creating ResNet18...")
    try:
        model = models.resnet18(weights=None)
        print("    Used weights=None (new API)")
    except TypeError:
        model = models.resnet18(pretrained=False)
        print("    Used pretrained=False (old API)")
    
    # Step 3: Modify FC layer
    print("\n[3] Modifying FC layer...")
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    
    # Step 4: Load weights
    print("\n[4] Loading state dict...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("    SUCCESS! Model loaded and ready.")
    
    # Step 5: Test inference
    print("\n[5] Testing inference...")
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    
    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(dummy).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    
    classes = checkpoint.get('class_names', ['fresh', 'rotten'])
    print(f"    Prediction: {classes[pred.item()]}")
    print(f"    Confidence: {conf.item() * 100:.1f}%")
    print("\n✅ ALL TESTS PASSED!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
