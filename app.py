# Streamlit app for SMS Spam Classification
# This app loads a trained ML model and vectorizer
# and predicts whether an input message is spam or not

import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", 'rb'))

# Load the text vectorizer used during training
cv = pickle.load(open("vectorizer.pkl", 'rb'))

# App title and description
st.title("Email Spam Classification application")
st.write("This is a Machine Learning application to classify emails as spam or ham.")

# User input
user_input = st.text_area("Enter an email to classify",height=150)

# Prediction button Logic
if st.button("Classify"):
    if user_input:
        # Convert input text into numerical features
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()

        # Predict using trained model
        result = model.predict(vectorized_data)

        # Display result
        if result[0]==0:
            st.write("The email is not spam")
        else:
            st.write("The email is spam")
    else:
        st.write("Please type email to classify")        
