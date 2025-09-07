import streamlit as st
import pandas as pd
import os
import hashlib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# User Database
# -----------------------------
USER_DB = "users.csv"

if not os.path.exists(USER_DB):
    df = pd.DataFrame(columns=["username", "password", "role"])
    df.to_csv(USER_DB, index=False)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    return pd.read_csv(USER_DB)

def save_user(username, password, role):
    df = load_users()
    if username in df["username"].values:
        return False
    hashed_pw = hash_password(password)
    new_user = pd.DataFrame([[username, hashed_pw, role]], columns=["username", "password", "role"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_DB, index=False)
    return True

def verify_user(username, password):
    df = load_users()
    hashed_pw = hash_password(password)
    user = df[(df["username"] == username) & (df["password"] == hashed_pw)]
    return not user.empty, user

# -----------------------------
# Load Disease Dataset & Model
# -----------------------------
@st.cache_data
def load_data():
    train_df = pd.read_csv("Training.csv")
    test_df = pd.read_csv("Testing.csv")
    return train_df, test_df

train_df, test_df = load_data()

X = train_df.drop("prognosis", axis=1)
y = train_df["prognosis"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_val, model.predict(X_val))

# Disease knowledge base
disease_info = {
    "Foot and Mouth Disease": {
        "Description": "A severe viral disease affecting cloven-hoofed animals.",
        "Causes": "Caused by the Foot-and-Mouth Disease Virus (FMDV).",
        "Prevention": "Vaccination, quarantine, strict biosecurity.",
        "Treatment": "No direct cure, supportive care, antibiotics for infections."
    },
    "Mastitis": {
        "Description": "Inflammation of the udder tissue caused by bacteria.",
        "Causes": "Poor hygiene, injuries, bacterial contamination.",
        "Prevention": "Clean milking practices, disinfecting teats.",
        "Treatment": "Antibiotics, anti-inflammatory drugs, frequent milking."
    },
    # Extend dictionary with more diseases
}

# -----------------------------
# Session State
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

# -----------------------------
# Pages
# -----------------------------
def register_page():
    st.markdown("<h2 style='color:green;text-align:center;'>📝 Register</h2>", unsafe_allow_html=True)
    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")
    role = st.selectbox("Select Role", ["Farmer", "Vet Doctor", "Admin"])
    if st.button("Register"):
        if username and password:
            success = save_user(username, password, role)
            if success:
                st.success("✅ Registration successful! Please login.")
            else:
                st.error("❌ Username already exists.")
        else:
            st.error("⚠️ Enter all fields.")

def login_page():
    st.markdown("<h2 style='color:blue;text-align:center;'>🔑 Login</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        valid, user = verify_user(username, password)
        if valid:
            st.session_state.logged_in = True
            st.session_state.role = user.iloc[0]["role"]
            st.session_state.username = username
            st.success(f"✅ Welcome {st.session_state.role} {username}!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")

# -----------------------------
# Dashboards
# -----------------------------
def farmer_dashboard():
    st.markdown("<h2 style='color:orange;'>👩‍🌾 Farmer Dashboard</h2>", unsafe_allow_html=True)
    st.write("Farmers can view general cattle records and milk yield insights.")
    st.dataframe(train_df.head())
    st.bar_chart(train_df.iloc[:, 1:5])

def vet_dashboard():
    st.markdown("<h2 style='color:purple;'>🐄 Vet Doctor Dashboard</h2>", unsafe_allow_html=True)
    st.sidebar.success(f"Model Accuracy: {acc*100:.2f}%")

    st.subheader("Select the Symptoms:")
    symptoms = list(X.columns)
    selected_symptoms = st.multiselect("Choose observed symptoms:", symptoms)

    if st.button("🔍 Predict Disease"):
        if selected_symptoms:
            input_data = [0] * len(symptoms)
            for s in selected_symptoms:
                input_data[symptoms.index(s)] = 1
            input_array = np.array(input_data).reshape(1, -1)

            prediction = model.predict(input_array)
            disease = le.inverse_transform(prediction)[0]

            st.success(f"✅ Predicted Disease: **{disease}**")

            if disease in disease_info:
                st.subheader("📖 Disease Information")
                st.write(f"**Description:** {disease_info[disease]['Description']}")
                st.write(f"**Causes:** {disease_info[disease]['Causes']}")
                st.write(f"**Prevention:** {disease_info[disease]['Prevention']}")
                st.write(f"**Treatment:** {disease_info[disease]['Treatment']}")
            else:
                st.warning("ℹ️ Detailed info not available. Please consult a veterinarian.")
        else:
            st.warning("⚠️ Please select at least one symptom.")

def admin_dashboard():
    st.markdown("<h2 style='color:red;'>🛡️ Admin Dashboard</h2>", unsafe_allow_html=True)
    st.write("Admins can monitor users & system status.")
    st.dataframe(load_users())

# -----------------------------
# Main App
# -----------------------------
def main():
    st.sidebar.title("🌾 AgroStock Navigation")

    if not st.session_state.logged_in:
        choice = st.sidebar.radio("Select", ["Login", "Register"])
        if choice == "Login":
            login_page()
        else:
            register_page()
    else:
        st.sidebar.write(f"Logged in as **{st.session_state.role} - {st.session_state.username}**")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.username = None
            st.rerun()

        if st.session_state.role == "Farmer":
            farmer_dashboard()
        elif st.session_state.role == "Vet Doctor":
            vet_dashboard()
        elif st.session_state.role == "Admin":
            admin_dashboard()

if __name__ == "__main__":
    main()
