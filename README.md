# 🏨 Hotel Booking ML Dashboard

A complete Machine Learning Data Analysis and Classification Dashboard built using Python, Pandas, Scikit-learn, and Streamlit.

This project analyzes hotel booking data, performs data cleaning and preprocessing, visualizes important patterns, trains classification models, makes predictions, and evaluates model performance through an interactive Streamlit dashboard.

## 🚀 Live Demo

🔗 **Streamlit App:**  
https://hotel-booking-ml-dashboard-2dprtqj43rj84aoxmqfrpv.streamlit.app/

## 📌 Project Overview

The Hotel Booking ML Dashboard provides an interactive interface for performing different stages of a Machine Learning workflow on the Hotel Booking Demand dataset.

The dashboard contains 10 experiments covering:

- Dataset loading
- Statistical analysis
- Missing value handling
- Data preprocessing
- Encoding and scaling
- Data visualization
- Feature representation
- Model selection
- Train-test splitting
- Model training
- Prediction
- Performance evaluation
- Confusion matrix
- Cleaned dataset download

## 🧪 Experiments

### 1. 📂 Dataset Loading
Loads and displays the hotel booking dataset along with the number of records and features.

### 2. 📊 Statistical Information
Displays:
- First five records
- Dataset shape
- Column information
- Data types
- Missing value counts
- Statistical description
- Categorical features

### 3. 🧹 Missing Values & Cleaning
Identifies missing values and cleans the dataset by removing unwanted columns and missing records.

### 4. ⚙️ Encoding & Scaling
Performs:
- One-Hot Encoding
- Standardization using StandardScaler
- Normalization using MinMaxScaler

### 5. 📈 Data Visualization
Visualizes:
- Hotel type distribution
- Booking cancellation distribution
- Average daily rate
- Average lead time

### 6. 🧩 Feature Representation
Selects important features and represents them as:
- Feature matrix X
- Target variable y
- Feature statistics

### 7. 🤖 Model Selection & Data Split
Provides classification model options:
- Logistic Regression
- Decision Tree
- Random Forest

The dataset is divided into training and testing data using an 80:20 split.

### 8. 🧠 Model Training
Trains the selected classification model using the training dataset.

### 9. 🔮 Prediction
Demonstrates:
- Prediction using labelled test data
- Prediction using new/unlabelled data

### 10. 📊 Performance Metrics
Evaluates the trained model using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Git
- GitHub

## 📁 Project Structure

```text
Hotel_Booking_ML_Project/
│
├── app.py
├── requirements.txt
│
└── dataset/
    └── hotel_bookings.csv
