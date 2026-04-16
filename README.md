# 🌿 Crop Disease Detection System
**Musinguzi Dickson, Muhairwe Dominic, Kalyegira Emmanuel, Emelda Nakacwa**  
*Group 2*
---

## System Overview

We developed an AI-powered system that allows farmers to photograph a crop leaf and receive an instant disease diagnosis with treatment recommendations. It is built using a Convolutional Neural Network (CNN) with MobileNetV2 transfer learning, trained on the PlantVillage dataset that we got from kaggle.

**AI Approach:** Computer Vision — Transfer Learning (MobileNetV2 + Custom Classification Head)  
**Language:** Python 3.10+  
**Frameworks Used:** TensorFlow / Keras, Streamlit for the UI

---

## Dataset

**PlantVillageDataset**  
- **Source:** https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset  
- **Size:** ~54,000 images across 38 disease/healthy classes  
- **Crops:** Tomato, Maize, Cassava, Potato, Pepper, Apple, Grape, etc 
- **Format:** RGB JPEG images, organized in folders by class name  
- 

### How to Download the Dataset
1. Create a free account at [https://www.kaggle.com](https://www.kaggle.com)
2. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
3. Click **Download** — extract the zip
4. Place the extracted folder at: `dataset/plantvillage/`

Your folder structure should look like:
```
dataset/
  plantvillage/
    Tomato___Late_blight/
      image001.jpg
      image002.jpg
      ...
    Tomato___Early_blight/
    Corn_(maize)___Common_rust_/
    Cassava___Bacterial_Blight/
    ... (38 folders total)
```

---

## Prerequisites

- Python 3.10 or higher  
- pip package manager  
- At least 4GB RAM  
- GPU recommended but not required (CPU training takes longer though)

---

## Installation & Setup

### Step 1 — Clone / Download the Project
```bash
# If you are using git
git clone <your-repo-url>
cd crop_disease_project

# Or unzip the submitted folder
cd crop_disease_project
```

### Step 2 — Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download Dataset
Follow the dataset instructions above. Place data at `dataset/plantvillage/`.

---

## Running the System

### Step 1 — Train the Model
```bash
python src/train_model.py
```
This:
- Loads training images
- Trains MobileNetV2 on your dataset
- Saves the model to `models/crop_disease_model.h5`
- Saves class labels to `models/class_labels.json`
- Saves training curves to `results/training_curves.png`

*Training time: 15–45 minutes on CPU/ depending on your laptop speed, 5 minutes on GPU*

### Step 2 — Evaluate the Model
```bash
python src/evaluate_model.py
```
This:
- Computes accuracy, precision, recall, F1 score
- Generates confusion matrix → `results/confusion_matrix.png`
- Identifies top failure cases → `results/failure_cases.json`
- Saves full report → `results/classification_report.txt`

### Step 3 — Run the Web Application
```bash
streamlit run app.py
```
Then open your browser at: **http://localhost:8501**

Upload any plant leaf photo to receive an instant AI diagnosis.

---

## Input and Output

### Input
- A photograph of a plant leaf (JPG or PNG format)
- Can be taken with a smartphone camera
- Works best with: clear focus, good lighting, single affected leaf

### Output
- **Predicted disease** (e.g., "Tomato — Late Blight")
- **Confidence score** (e.g., 87.3%)
- **Disease description**
- **Treatment recommendation**
- **Prevention tips**
- **Top-3 alternative diagnoses**

---

## Project File Structure

```
crop_disease_project/
│
├── app.py                    ← Streamlit web application (run this for demo)
│
├── src/
│   ├── train_model.py        ← CNN training script
│   └── evaluate_model.py     ← Evaluation & metrics script
│
├── models/                   ← Created after training
│   ├── crop_disease_model.h5 ← Saved trained model
│   └── class_labels.json     ← Disease class index mapping
│
├── dataset/
│   └── plantvillage/         ← Place downloaded dataset here
│
├── results/                  ← Created after evaluation
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── failure_cases.json
│
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## Dependencies (`requirements.txt`)

```
tensorflow>=2.12.0
streamlit>=1.28.0
numpy>=1.23.0
Pillow>=9.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
```



