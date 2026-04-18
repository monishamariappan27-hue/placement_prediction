import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Placement Prediction", layout="wide")

# -----------------------------
# THEME TOGGLE (Dark / Light)
# -----------------------------
theme = st.sidebar.radio("🎨 Select Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        body {background-color: #0E1117; color: white;}
        </style>
    """, unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🎓 Student Placement Prediction Dashboard")
st.markdown("### Predict whether a student will be **Placed or Not Placed**")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("placement.csv")

    df["PlacementStatus"] = df["PlacementStatus"].map({
        "Placed": 1,
        "NotPlaced": 0
    })

    df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({
        "Yes": 1,
        "No": 0
    })

    df["PlacementTraining"] = df["PlacementTraining"].map({
        "Yes": 1,
        "No": 0
    })

    return df

df = load_data()

# -----------------------------
# MODEL TRAINING
# -----------------------------
X = df.drop(["PlacementStatus", "StudentID"], axis=1)
y = df["PlacementStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -----------------------------
# METRICS
# -----------------------------
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy*100:.2f}%")
col2.metric("Total Students", len(df))
col3.metric("Features Used", len(X.columns))

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("📥 Student Input")

cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)
internships = st.sidebar.number_input("Internships", 0, 10, 1)
projects = st.sidebar.number_input("Projects", 0, 10, 2)
workshops = st.sidebar.number_input("Workshops", 0, 10, 1)
aptitude = st.sidebar.slider("Aptitude Score", 0, 100, 60)
softskills = st.sidebar.slider("Soft Skills", 0.0, 5.0, 3.0)

extra = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
training = st.sidebar.selectbox("Placement Training", ["Yes", "No"])

ssc = st.sidebar.slider("SSC Marks", 0, 100, 70)
hsc = st.sidebar.slider("HSC Marks", 0, 100, 75)

extra = 1 if extra == "Yes" else 0
training = 1 if training == "Yes" else 0

# -----------------------------
# PREDICTION
# -----------------------------
st.subheader("🤖 Prediction")

input_data = pd.DataFrame([[cgpa, internships, projects, workshops,
                            aptitude, softskills, extra, training, ssc, hsc]],
                          columns=X.columns)

if st.button("🚀 Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Likely to be PLACED")
    else:
        st.error("⚠️ Likely NOT placed")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, model.predict(X_test))

fig, ax = plt.subplots()
ax.matshow(cm)

for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f"{val}", ha='center', va='center')

st.pyplot(fig)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
st.subheader("📄 Classification Report")

report = classification_report(y_test, model.predict(X_test), output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📈 Feature Importance")

importance = pd.Series(model.coef_[0], index=X.columns)
importance.sort_values().plot(kind='barh')

st.pyplot(plt)

# -----------------------------
# DATA VISUALIZATION
# -----------------------------
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("Placement Distribution")
    st.bar_chart(df["PlacementStatus"].value_counts())

with col2:
    st.write("CGPA Distribution")
    st.line_chart(df["CGPA"])

# -----------------------------
# DOWNLOAD PREDICTION
# -----------------------------
st.subheader("📥 Download Sample Input")

csv = input_data.to_csv(index=False)
st.download_button("Download Input Data", csv, "sample_input.csv")

# -----------------------------
# SHOW DATA
# -----------------------------
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head())

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("👩‍💻 Developed by Monisha | Placement Prediction Project")
