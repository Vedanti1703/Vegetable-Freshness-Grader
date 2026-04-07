# 🥦 Vegetable Freshness Grader - Custom CNN Version

## 📁 Files Included

1. **STEP_BY_STEP_GUIDE.md** - Detailed implementation instructions
2. **MIGRATION_GUIDE.md** - Complete migration documentation
3. **train_model.py** - Model training script
4. **pipeline_updated.py** - New pipeline (replaces old pipeline.py)
5. **requirements_updated.txt** - Updated dependencies
6. **index_updated.html** - Updated frontend (replaces frontend/index.html)
7. **setup.sh** - Automated setup script (Linux/Mac)

---

## ⚡ Quick Start

### 1. Download Dataset
```bash
# Get from Kaggle:
https://www.kaggle.com/datasets/muhriddinmuxiddinov/vegetables-fresh-and-rotten-for-classification

# Extract to: dataset/train/fresh/ and dataset/train/rotten/
```

### 2. Replace Files in Your Project
```bash
# Backup originals first!
cp pipeline.py pipeline_old.py
cp requirements.txt requirements_old.txt
cp frontend/index.html frontend/index_old.html

# Copy new files
cp requirements_updated.txt requirements.txt
cp pipeline_updated.py pipeline.py
cp index_updated.html frontend/index.html
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Model
```bash
python3 train_model.py
# Takes 15-30 minutes with GPU, 2-3 hours with CPU
# Creates freshness_model.pth (~45 MB)
```

### 5. Run Application
```bash
# Start backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Open frontend/index.html in browser
```

---

## 🔑 Key Changes

### What's Different:
- ❌ No Groq API required
- ✅ Custom trained CNN model
- ✅ 100% local/offline
- ✅ Faster inference (100-300ms)
- ✅ No API costs
- ✅ Same UI design

### Files Modified:
1. `pipeline.py` - Complete rewrite (no Groq)
2. `requirements.txt` - Added PyTorch, removed Groq
3. `frontend/index.html` - Only 2 text changes

### Files Added:
1. `train_model.py` - Train your own model
2. `freshness_model.pth` - Generated after training

---

## 📊 Expected Results

**Training:**
- Accuracy: 85-95%
- Training time: 15-30 minutes (GPU)
- Model size: ~45 MB

**Inference:**
- Speed: 100-300ms per image
- Works offline: Yes
- GPU recommended: Yes (but CPU works too)

---

## 📖 Documentation

- **For step-by-step instructions:** Read `STEP_BY_STEP_GUIDE.md`
- **For detailed explanation:** Read `MIGRATION_GUIDE.md`

---

## 🐛 Common Issues

**"Model file not found"**
→ Run `python3 train_model.py` first

**"CUDA out of memory"**
→ Reduce BATCH_SIZE in train_model.py

**"Dataset folder not found"**
→ Create dataset/train/fresh/ and dataset/train/rotten/

---

## ✅ Verification Checklist

- [ ] Dataset downloaded and organized
- [ ] Dependencies installed
- [ ] Model trained (freshness_model.pth exists)
- [ ] Backend starts without errors
- [ ] Frontend shows same UI
- [ ] Can analyze images successfully

---

## 🎯 Dataset Link

**Primary:**
https://www.kaggle.com/datasets/muhriddinmuxiddinov/vegetables-fresh-and-rotten-for-classification

**Alternative:**
https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

---

## 💡 Tips

1. Use GPU for training (15-30 min vs 2-3 hours)
2. Increase EPOCHS to 25-30 for better accuracy
3. Test with various vegetable images
4. Keep UI exactly the same - users won't notice backend change!

---

## 🚀 Ready to Start?

1. Read `STEP_BY_STEP_GUIDE.md` thoroughly
2. Download the dataset
3. Follow the 9 steps exactly
4. Test with sample images
5. Enjoy your API-free system!

**Good luck! 🎉**
