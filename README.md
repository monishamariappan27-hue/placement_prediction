# 🎓 Student Placement Prediction Dashboard

A Machine Learning powered **Streamlit web application** that predicts whether a student will be **Placed or Not Placed** based on academic performance, skills, and other features.

🌐 Live App: https://autonomous-vehicle-forapp87sg2pmx3s8w6u2dr.streamlit.app/

---

## 📌 Project Overview

This project uses **Logistic Regression** to analyze student data and predict placement outcomes. It also includes an interactive dashboard built using **Streamlit** for real-time predictions and data visualization.

---

## 🚀 Features

- 📊 Model Performance Metrics (Accuracy, Confusion Matrix)
- 🤖 Real-time Placement Prediction
- 🎛️ Interactive Sidebar Inputs
- 📈 Feature Importance Visualization
- 📉 Data Insights & Charts
- 🌗 Dark / Light Theme Toggle
- 📥 Download Input Data as CSV
- 📄 Classification Report

---

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression
- Library: Scikit-learn
- Train-Test Split: 80/20
- Output: Binary Classification (Placed / Not Placed)

---

## 📂 Dataset Features

The model uses the following features:

- CGPA
- Internships
- Projects
- Workshops
- Aptitude Score
- Soft Skills
- Extracurricular Activities
- Placement Training
- SSC Marks
- HSC Marks

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🎈
- Pandas 📊
- NumPy 🔢
- Scikit-learn 🤖
- Matplotlib 📉

---

## 📊 Project Workflow

1. Load dataset (`placement.csv`)
2. Preprocess data (encoding categorical values)
3. Train Logistic Regression model
4. Evaluate model performance
5. Build interactive Streamlit dashboard
6. Deploy web application

---

## 📸 Screenshots

*(Add screenshots of your Streamlit dashboard here)*

---

## ⚙️ How to Run Locally

```bash
# Clone repository
git clone https://github.com/your-username/placement-prediction.git

# Go to project folder
cd placement-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
