import streamlit as st
import joblib


model = joblib.load("model.pkl")
vectorizer = joblib.load("vect.pkl")


st.title("📝 Sentiment Analysis App")
st.write("Enter any sentence below.")


text = st.text_area("Enter Text")


if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        
        text_vector = vectorizer.transform([text])

        
        prediction = model.predict(text_vector)

        
        if prediction[0] == 1 or prediction[0] == "Positive":
            st.success("😊 Positive Sentiment")
        elif prediction[0] == 0 or prediction[0] == "Negative":
            st.error("😞 Negative Sentiment")
        else:
            st.info(f"Prediction: {prediction[0]}")