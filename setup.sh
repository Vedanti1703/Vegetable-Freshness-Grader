#!/bin/bash

# ════════════════════════════════════════════════════════════════
# Vegetable Freshness Grader - Quick Setup Script
# ════════════════════════════════════════════════════════════════

echo "🥦 Vegetable Freshness Grader - Setup"
echo "======================================"
echo ""

# Step 1: Check Python version
echo "1️⃣  Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo "✅ Python found"
echo ""

# Step 2: Install dependencies
echo "2️⃣  Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Step 3: Check for dataset
echo "3️⃣  Checking for dataset..."
if [ ! -d "dataset/train" ]; then
    echo "⚠️  Dataset not found!"
    echo ""
    echo "Please download the dataset from:"
    echo "https://www.kaggle.com/datasets/muhriddinmuxiddinov/vegetables-fresh-and-rotten-for-classification"
    echo ""
    echo "Extract it and organize as:"
    echo "  dataset/"
    echo "    └── train/"
    echo "        ├── fresh/"
    echo "        └── rotten/"
    echo ""
    exit 1
else
    echo "✅ Dataset found"
fi
echo ""

# Step 4: Check for trained model
echo "4️⃣  Checking for trained model..."
if [ ! -f "freshness_model.pth" ]; then
    echo "⚠️  Model not trained yet!"
    echo ""
    read -p "Do you want to train the model now? (y/n): " train_choice
    if [ "$train_choice" = "y" ]; then
        echo ""
        echo "Starting training... This will take 15-30 minutes."
        python3 train_model.py
        if [ $? -eq 0 ]; then
            echo "✅ Model trained successfully!"
        else
            echo "❌ Training failed. Check errors above."
            exit 1
        fi
    else
        echo "Please run 'python3 train_model.py' before starting the server."
        exit 1
    fi
else
    echo "✅ Trained model found"
fi
echo ""

# Step 5: Start the server
echo "5️⃣  Starting FastAPI server..."
echo ""
echo "Backend will run on: http://localhost:8000"
echo "Open frontend/index.html in your browser to use the app"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn main:app --reload --host 0.0.0.0 --port 8000
