import streamlit as st
import numpy as np
import pickle

# Load the saved model
model = pickle.load(open('trained_model.sav', 'rb'))

# Language setting
if "lang" not in st.session_state:
    st.session_state.lang = "en"

def toggle_language():
    st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"

# Language texts with decorations
texts = {
    "en": {
        "title": "🌸 Iris Flower Prediction System",
        "sepal_length": "Sepal Length",
        "sepal_width": "Sepal Width",
        "petal_length": "Petal Length",
        "petal_width": "Petal Width",
        "result": "Result",
        "predict_button": "Predict 🌺",
        "change_language": "Change Language 🌐",
        "success_message": "The predicted iris type is: "
    },
    "ar": {
        "title": "🌸 نظام التنبؤ بنوع زهرة السوسن",
        "sepal_length": "طول السبلة",
        "sepal_width": "عرض السبلة",
        "petal_length": "طول البتلة",
        "petal_width": "عرض البتلة",
        "result": "النتيجة",
        "predict_button": "تنبؤ 🌺",
        "change_language": "تغيير اللغة 🌐",
        "success_message": "نوع السوسن المتوقع هو: "
    }
}

# Streamlit app title with language toggle button
st.title(texts[st.session_state.lang]["title"])
st.button(texts[st.session_state.lang]["change_language"], on_click=toggle_language)

# Input fields
sepal_length = st.number_input(texts[st.session_state.lang]["sepal_length"], min_value=0.0, step=0.1)
sepal_width = st.number_input(texts[st.session_state.lang]["sepal_width"], min_value=0.0, step=0.1)
petal_length = st.number_input(texts[st.session_state.lang]["petal_length"], min_value=0.0, step=0.1)
petal_width = st.number_input(texts[st.session_state.lang]["petal_width"], min_value=0.0, step=0.1)

# Prepare input data for prediction
input_data = [sepal_length, sepal_width, petal_length, petal_width]

# Prediction button and result display
if st.button(texts[st.session_state.lang]["predict_button"]):
    input_data = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    result = prediction[0]
    st.success(f"{texts[st.session_state.lang]['success_message']} {result}")
