import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def display_no_depression_tips():
    st.title("No Depression Detected")
    st.write("It seems that no signs of depression were detected.")
    st.write("Here are some tips to maintain mental well-being:")
    st.write("- Practice mindfulness and relaxation techniques.")
    st.write("- Engage in regular physical activity.")
    st.write("- Stay connected with loved ones.")
    st.write("- Seek professional help if needed.")

def display_moderate_depression_tips():
    st.title("Moderate Depression Detected")
    st.write("It appears that signs of moderate depression were detected.")
    st.write("Here are some tips to help cope with depression:")
    st.write("- Radiant Resilience: 🌈 Embrace your inner superhero! You're navigating a challenging chapter with courage and resilience. Every step forward is a victory dance waiting to happen.")
    st.write("- Bold Self-Care Brushstrokes: 🎨 Paint your world with bold strokes of self-care. From cozy cups of tea to dancing like nobody's watching, indulge in moments that whisper")
    st.write("- Vibrant Venting Sessions: 💬 Unleash your thoughts! Whether it's with a trusted friend, a journal, or a therapeutic howl to the moon, give your feelings the freedom to breathe.")
    st.write("- Sunshine Seeking: Chase the sun! Even if it's just a brief stroll, let the warmth of sunlight touch your face. Nature's embrace is a vibrant reminder that brighter days await.")

def display_severe_depression_tips():
    st.title("Severe Depression Detected")
    st.write("It seems that signs of severe depression were detected.")
    st.write("It is important to seek professional help immediately.")
    st.write("Here are some steps to take:")
    st.write("- Contact a mental health professional or therapist.")
    st.write("- Reach out to a trusted friend or family member.")
    st.write("- Consider helplines or support groups for assistance.")

# Load the dataset
f = "ML-DataSet_5.csv"
df = pd.read_csv(f)
df1 = df.drop(['f_id', 'duplicate_x', 'duplicate_y', 'duplicate_z', 'duplicate_v', 'duplicate_w', 
               'duplicate_a', 'dup_b', 'dup_c', 'Anxeity_Rec'], axis=1)
df1.fillna(df1.mean(), inplace=True)

# Selecting columns for Label Encoding
columns_to_encode = ['part1_country', 'Locality_first', 'part1_current_preg_first', 'anxious', 'Anxietyrec2',
                     'worry', 'relaxing', 'restless', 'annoyed', 'afraid', 'interest', 'hopeless',
                     'sleep cycle', 'tiredness', 'appetite', 'regret', 'focus', 'isolated', 'pessimism',
                     'AnxietyCat', 'DepressionCat', 'WomenAge', 'MariageAge', 'NumberofPregnancy',
                     'NumberofAbortions', 'EducationLevel', 'Work', 'CovidDiagL1', 'Health_Prob', 
                     'FamilyProblems', 'FinancialProblem', 'SocialProblem']

# Encode categorical features
label_encoders = {}
for col in columns_to_encode:
    le = LabelEncoder()
    df1[col] = le.fit_transform(df1[col].astype(str))
    label_encoders[col] = le

# Split the data
X = df1.drop('DepressionCat', axis=1)  # Replace 'DepressionCat' with your target column
y = df1['DepressionCat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit app interface
st.title("Depression Severity Detection App")

# Logo and description
st.image("1223.png", width=100)  # Replace with the path to your logo image
st.write("""
This app helps in detecting the severity of depression based on user inputs.
It leverages a machine learning model trained on various factors associated with mental health.
Use the sidebar to input your details and check your results.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Features")
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.text_input(f"Enter {col}:", "")

# Convert input to a DataFrame
input_df = pd.DataFrame([input_data])

# Encode the input data
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col].astype(str))

# Standardize the input data
input_df = scaler.transform(input_df)

# Make predictions
if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 0:  # Assuming 0 means no depression
        display_no_depression_tips()
    elif prediction[0] == 1:  # Assuming 1 means moderate depression
        display_moderate_depression_tips()
    else:  # Assuming 2 means severe depression
        display_severe_depression_tips()
