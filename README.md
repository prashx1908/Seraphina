# Seraphina: Your Insight Partner

Seraphina is a Post Partum Depression analysis platform designed to help users assess their mental well-being and provide tips for improving it. By leveraging machine learning techniques, Seraphina detects signs of depression and offers personalized advice based on the severity of the detected condition.

## Table of Contents

- [Introduction](#introduction)
- [Site Link](#SiteLink)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Mental health is an important aspect of overall well-being. Seraphina is a tool that aims to provide insights into a user's mental state by analyzing various factors and symptoms. It provides tailored tips and resources based on the results of the analysis, which can help users take steps toward better mental health.

## Site Link
- **Seraphina**:[Click Here](https://seraphina.streamlit.app/)

## Features

- **Depression Detection**: Predicts the likelihood of depression based on user input.
- **Personalized Tips**: Offers mental health tips and advice tailored to the detected level of depression (No, Moderate, or Severe).
- **Interactive Interface**: User-friendly interface built with Streamlit.
- **Machine Learning Model**: Uses a Random Forest Classifier for prediction.
- **Real-Time Predictions**: Get predictions instantly after providing the necessary inputs.

## Technologies Used

- **Python**
- **Streamlit**: Frontend interface for user interaction.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Scikit-learn**: For building and training the machine learning model.
- **Random Forest Classifier**: The core model used for depression prediction.

## Installation

To get started with Seraphina, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/prashx1908/Seraphina.git
    cd seraphina
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**
    ```bash
    streamlit run app.py
   
