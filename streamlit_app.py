import streamlit as st
import numpy as np
import pickle

# loading the saved model
model = pickle.load(open('trained_model.sav', 'rb'))


##streamlit app
st.title('Iris Prediction')

sepal_length = st.number_input('Sepal Length')
sepal_width = st.number_input('Sepal Width')
petal_length = st.number_input('Petal Length')
petal_width = st.number_input('Petal Width')


input_data = [sepal_length, sepal_width, petal_length, petal_width]


# code for Prediction
result = ''

# creating a button for Prediction

if st.button('Result'):
  input_data = np.asarray(input_data).reshape(1,-1)
  prediction = model.predict(input_data)
  result = prediction[0]


st.success(result)
