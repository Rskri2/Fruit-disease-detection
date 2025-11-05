# Machine Learning Models with Flask Deployment

## Overview

This repository contains multiple **machine learning and deep learning models** for classification tasks, including:

- **Cubic SVM** (Polynomial kernel, degree=3)
- **Fine KNN** (K-Nearest Neighbors)
- **Bagged Tree** (Bagging ensemble of decision trees)
- **Boosted Tree** (Gradient Boosted Trees)
- **Complex Tree** (Deep Decision Tree)
- **CNN from Scratch** (Convolutional Neural Network)
- **VGG16** (Pretrained convolutional network)
- **ResNet** (Pretrained residual network)

The models are trained on a **custom dataset** and can be deployed via a **Flask web application** for real-time inference.

---

## Repository Structure
```
├── data   
│   ├── train      
│   ├── test      
│   └── validation  
├── notebooks
│   └── Bagged_Tree.ipynb  
│   └── Boosted_Tree.ipynb  
│   └── CNN.ipynb  
│   └── Complex_Tree.ipynb  
│   └── Cubic_SVM.ipynb  
│   └── Fine_KNN.ipynb  
│   └── RESNET.ipynb  
│   └── VGG16.ipynb 
├── app         
│   ├── app.py          
│   ├── templates
│   │   └── index.html  
│   │   └── diesease.html 
│   └── model.h5         
├── requirements.txt    
├── .gitignore   
└── README.md           
```
---

## Dataset

- The dataset is stored in the `data/` folder and is split into **training**, **testing**, and  **validation** sets.
```
data
└── train
    ├── class1
    │   ├── img1.jpg
    │   └── img2.jpg
    └── class2
        ├── img1.jpg
        └── img2.jpg
└── test
└── validation
```
## Installation

1. Clone the repository:

```bash
 git clone https://github.com/Rskri2/Fruit-disease-detection.git  
cd Fruit-disease-detection
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Jupyter Notebooks

1. Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

2. Launch Jupyter Notebook:

```bash
jupyter notebook
```
3. Open the notebook file notebooks/model_training.ipynb in your browser.

4. Execute the cells sequentially to evaluate models

5. Save trained models in .h5 file

## Flask Deployment

The Flask app allows users to upload data or input features and get predictions from the trained model.

1. Run Flask app:
```bash
    cd app
    python app.py
```
2. Visit http://127.0.0.1:5000 in your browser.

3. Upload an image or enter features to get predictions.