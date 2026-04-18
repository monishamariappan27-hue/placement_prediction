import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Placement Prediction", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("🎓 Placement Prediction System")

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
# MODEL
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
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "📁 Data"])

# =============================
# TAB 1: DASHBOARD
# =============================
with tab1:
    st.subheader("📊 Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Students", len(df))
    col3.metric("Features", len(X.columns))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Placement Distribution")
        st.bar_chart(df["PlacementStatus"].value_counts())

    with col2:
        st.write("CGPA Distribution")
        st.line_chart(df["CGPA"])

    st.markdown("---")

    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(y_test, model.predict(X_test))

    fig, ax = plt.subplots()
    ax.matshow(cm)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha='center', va='center')

    st.pyplot(fig)

# =============================
# TAB 2: PREDICTION
# =============================
with tab2:
    st.subheader("🤖 Predict Placement")

    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
        internships = st.number_input("Internships", 0, 10, 1)
        projects = st.number_input("Projects", 0, 10, 2)
        workshops = st.number_input("Workshops", 0, 10, 1)
        aptitude = st.slider("Aptitude Score", 0, 100, 60)

    with col2:
        softskills = st.slider("Soft Skills", 0.0, 5.0, 3.0)
        extra = st.selectbox("Extracurricular", ["Yes", "No"])
        training = st.selectbox("Training", ["Yes", "No"])
        ssc = st.slider("SSC Marks", 0, 100, 70)
        hsc = st.slider("HSC Marks", 0, 100, 75)

    extra = 1 if extra == "Yes" else 0
    training = 1 if training == "Yes" else 0

    input_data = pd.DataFrame([[cgpa, internships, projects, workshops,
                                aptitude, softskills, extra, training, ssc, hsc]],
                              columns=X.columns)

    if st.button("🚀 Predict"):
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("🎉 Likely to be PLACED")
        else:
            st.error("⚠️ Likely NOT placed")

# =============================
# TAB 3: DATA
# =============================
with tab3:
    st.subheader("📁 Dataset")

    st.dataframe(df.head())

    st.download_button(
        "Download Dataset",
        df.to_csv(index=False),
        "placement.csv"
    )
