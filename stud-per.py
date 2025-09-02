import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder,StandardScaler

def load_model():
    with open("student_lr_final_model.pkl", "rb") as file:
        model,scaler,le = pickle.load(file)
    return model, scaler, le

def preprocession_input_data(data, scaler, le):
    # Convert single value to array for LabelEncoder
    extracurricular_value = [data["Extracurricular Activities"]]
    data["Extracurricular Activities"] = le.transform(extracurricular_value)[0]
    
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocession_input_data(data, scaler, le)
    prediction = model.predict(processed_data)  # Fixed typo: ppredict -> predict
    return prediction[0]  # Return single value instead of array

def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")

    hour_studied = st.number_input("Hours Studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("Previous Score", min_value=40, max_value=100, value=70)
    extra = st.selectbox("Extra Curricular Activity", ["Yes", "No"])
    sleeping_hour = st.number_input("Sleeping Hours", min_value=4, max_value=10, value=7)
    number_of_papers_solved = st.number_input("Number of Question Papers Solved", min_value=0, max_value=10, value=5)
    
    if st.button("Predict Your Score"):
        user_data = {
            "Hours Studied" : hour_studied,
            "Previous Scores" : previous_score,
            "Extracurricular Activities" : extra,
            "Sleep Hours" : sleeping_hour,
            "Sample Question Papers Practiced" : number_of_papers_solved,
        }
        prediction = predict_data(user_data)
        st.success(f"Your predicted performance score is: {prediction:.2f}")

if __name__ == "__main__":
    main()