import streamlit as st
import pandas as pd

# Page settings
st.set_page_config(
    page_title="Hotel Booking ML Dashboard",
    page_icon="🏨",
    layout="wide"
)

# Title
st.title("🏨 Hotel Booking ML Dashboard")
st.write("Machine Learning Data Analysis and Classification")

# Sidebar
st.sidebar.title("📌 Experiments")

experiment = st.sidebar.radio(
    "Select Experiment",
    [
        "Experiment 1 - Dataset Loading",
        "Experiment 2 - Statistical Information",
        "Experiment 3 - Missing Values & Cleaning",
        "Experiment 4 - Encoding & Scaling",
        "Experiment 5 - Visualization",
        "Experiment 6 - Feature Representation",
        "Experiment 7 - Model Selection & Data Split",
        "Experiment 8 - Model Training",
        "Experiment 9 - Prediction",
        "Experiment 10 - Performance Metrics",
        "📥 Download Cleaned Dataset"
    ]
)

# Load dataset
df = pd.read_csv("dataset/hotel_bookings.csv")

# Experiment 1
if experiment == "Experiment 1 - Dataset Loading":

    st.header("📂 Experiment 1 - Dataset Loading")

    st.write(
        "This experiment loads the Hotel Booking dataset "
        "and displays the complete dataset."
    )

    st.success("✅ Dataset loaded successfully!")

    # Dataset information
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", df.shape[0])

    with col2:
        st.metric("Total Features", df.shape[1])

    st.subheader("📋 Complete Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        height=600
    )
    
# Experiment 2
elif experiment == "Experiment 2 - Statistical Information":

    st.header("📊 Experiment 2 - Statistical Information")

    st.write(
        "This experiment provides basic structure and "
        "statistical information about the dataset."
    )

    # 1. First 5 rows
    st.subheader("1️⃣ First 5 Rows - head()")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # 2. Dataset Shape
    st.subheader("2️⃣ Dataset Shape")

    rows, columns = df.shape

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", rows)

    with col2:
        st.metric("Columns", columns)

    # 3. Dataset Information
    st.subheader("3️⃣ Dataset Information - info()")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    # 4. Statistical Description
    st.subheader("4️⃣ Statistical Description - describe()")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    # 5. Categorical Information
    st.subheader("5️⃣ Categorical Features")

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    st.write(
        "Number of categorical columns:",
        len(categorical_columns)
    )

    st.write(
        "Categorical columns:",
        list(categorical_columns)
    )
    
# Experiment 3
elif experiment == "Experiment 3 - Missing Values & Cleaning":

    st.header("🧹 Experiment 3 - Missing Values & Cleaning")

    st.write(
        "This experiment identifies missing values and "
        "cleans the dataset using column removal and dropna()."
    )

    # Missing value count
    st.subheader("1️⃣ Missing Value Count")

    missing_count = df.isnull().sum()

    missing_table = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_count.values
    })

    missing_table = missing_table[
        missing_table["Missing Values"] > 0
    ].sort_values(
        "Missing Values",
        ascending=False
    )

    st.dataframe(
        missing_table,
        use_container_width=True
    )

    # Missing percentage
    st.subheader("2️⃣ Missing Value Percentage")

    missing_percentage = (
        df.isnull().mean() * 100
    ).round(2)

    percentage_table = pd.DataFrame({
        "Column": df.columns,
        "Missing Percentage": missing_percentage.values
    })

    percentage_table = percentage_table[
        percentage_table["Missing Percentage"] > 0
    ].sort_values(
        "Missing Percentage",
        ascending=False
    )

    st.dataframe(
        percentage_table,
        use_container_width=True
    )

    # Before cleaning
    st.subheader("3️⃣ Dataset Before Cleaning")

    before_rows, before_columns = df.shape

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows Before Cleaning", before_rows)

    with col2:
        st.metric("Columns Before Cleaning", before_columns)

    # Remove highly missing column
    cleaned_df = df.drop(
        columns=["company"]
    )

    # Remove remaining rows containing null values
    cleaned_df = cleaned_df.dropna()

    # After cleaning
    st.subheader("4️⃣ Dataset After Cleaning")

    after_rows, after_columns = cleaned_df.shape

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows After Cleaning", after_rows)

    with col2:
        st.metric("Columns After Cleaning", after_columns)

    st.success("✅ Missing values removed successfully!")

    # Final null count
    st.subheader("5️⃣ Final Null Value Check")

    remaining_nulls = cleaned_df.isnull().sum().sum()

    st.metric(
        "Total Remaining Null Values",
        remaining_nulls
    )

    if remaining_nulls == 0:
        st.success("🎉 Dataset is completely clean!")
    else:
        st.warning("Some missing values are still present.")

    # Display cleaned dataset
    st.subheader("6️⃣ Cleaned Dataset")

    st.dataframe(
        cleaned_df.head(10),
        use_container_width=True
    )
    
# Experiment 4
elif experiment == "Experiment 4 - Encoding & Scaling":

    st.header("⚙️ Experiment 4 - Encoding & Scaling")

    st.write(
        "This experiment demonstrates categorical encoding, "
        "standardization, and normalization."
    )

    # Create cleaned dataset
    cleaned_df = df.drop(columns=["company"]).dropna()

    # ------------------------------------------------
    # 1. Encoding
    # ------------------------------------------------

    st.subheader("1️⃣ Categorical Encoding")

    st.write(
        "Categorical columns are converted into numerical "
        "form using One-Hot Encoding."
    )

    categorical_columns = cleaned_df.select_dtypes(
        include="object"
    ).columns

    st.write(
        "Categorical Columns:",
        list(categorical_columns)
    )

    encoded_df = pd.get_dummies(
        cleaned_df,
        columns=categorical_columns,
        drop_first=True
    )

    st.write(
        "Shape Before Encoding:",
        cleaned_df.shape
    )

    st.write(
        "Shape After Encoding:",
        encoded_df.shape
    )

    st.subheader("Encoded Dataset")

    st.dataframe(
        encoded_df.head(),
        use_container_width=True
    )

    # ------------------------------------------------
    # 2. Standardization
    # ------------------------------------------------

    st.subheader("2️⃣ Standardization")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    numeric_columns = cleaned_df.select_dtypes(
        include="number"
    ).columns

    standardized_data = scaler.fit_transform(
        cleaned_df[numeric_columns]
    )

    standardized_df = pd.DataFrame(
        standardized_data,
        columns=numeric_columns
    )

    st.write(
        "Standardization converts numerical features "
        "to approximately mean = 0 and standard deviation = 1."
    )

    st.dataframe(
        standardized_df.head(),
        use_container_width=True
    )

    # ------------------------------------------------
    # 3. Normalization
    # ------------------------------------------------

    st.subheader("3️⃣ Normalization")

    from sklearn.preprocessing import MinMaxScaler

    normalizer = MinMaxScaler()

    normalized_data = normalizer.fit_transform(
        cleaned_df[numeric_columns]
    )

    normalized_df = pd.DataFrame(
        normalized_data,
        columns=numeric_columns
    )

    st.write(
        "Normalization scales numerical values "
        "between 0 and 1."
    )

    st.dataframe(
        normalized_df.head(),
        use_container_width=True
    )

    st.success(
        "✅ Encoding, Standardization and Normalization completed!"
    )
    
# Experiment 5
elif experiment == "Experiment 5 - Visualization":

    st.header("📊 Experiment 5 - Data Visualization")

    st.write(
        "This experiment visualizes important patterns "
        "and distributions in the hotel booking dataset."
    )

    # Create cleaned dataset
    cleaned_df = df.drop(columns=["company"]).dropna()

    # -----------------------------------------
    # 1. Hotel Type Distribution
    # -----------------------------------------

    st.subheader("1️⃣ Hotel Type Distribution")

    hotel_counts = cleaned_df["hotel"].value_counts()

    st.bar_chart(hotel_counts)

    # -----------------------------------------
    # 2. Cancellation Distribution
    # -----------------------------------------

    st.subheader("2️⃣ Booking Cancellation Distribution")

    cancellation_counts = cleaned_df[
        "is_canceled"
    ].value_counts()

    cancellation_counts.index = [
        "Not Cancelled" if x == 0 else "Cancelled"
        for x in cancellation_counts.index
    ]

    st.bar_chart(cancellation_counts)

    # -----------------------------------------
    # 3. Average Daily Rate
    # -----------------------------------------

    st.subheader("3️⃣ Average Daily Rate by Hotel Type")

    adr_data = (
        cleaned_df
        .groupby("hotel")["adr"]
        .mean()
        .round(2)
    )

    st.bar_chart(adr_data)

    # -----------------------------------------
    # 4. Lead Time Distribution
    # -----------------------------------------

    st.subheader("4️⃣ Average Lead Time by Hotel Type")

    lead_time_data = (
        cleaned_df
        .groupby("hotel")["lead_time"]
        .mean()
        .round(2)
    )

    st.bar_chart(lead_time_data)

    st.success(
        "✅ Visualization completed successfully!"
    )
    
# -----------------------------------------
# Experiment 6 - Feature Representation
# -----------------------------------------

elif experiment == "Experiment 6 - Feature Representation":

    st.header("🧩 Experiment 6 - Feature Representation")

    st.write(
        "In this experiment, important features are selected "
        "and represented as input features for machine learning."
    )

    # Load and clean dataset
    cleaned_df = df.drop(columns=["company"]).dropna()

    # Select important numerical features
    selected_features = [
        "lead_time",
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "is_repeated_guest",
        "previous_cancellations",
        "booking_changes",
        "days_in_waiting_list",
        "adr"
    ]

    # Keep only available columns
    selected_features = [
        col for col in selected_features
        if col in cleaned_df.columns
    ]

    X = cleaned_df[selected_features]
    y = cleaned_df["is_canceled"]

    # -----------------------------------------
    # Selected Features
    # -----------------------------------------

    st.subheader("1️⃣ Selected Features")

    st.write(
        "Number of selected features:",
        len(selected_features)
    )

    st.write(selected_features)

    # -----------------------------------------
    # Feature Matrix
    # -----------------------------------------

    st.subheader("2️⃣ Feature Matrix (X)")

    st.write(
        "X represents the input features used by the ML model."
    )

    st.dataframe(
        X.head(10),
        use_container_width=True
    )

    # -----------------------------------------
    # Target Variable
    # -----------------------------------------

    st.subheader("3️⃣ Target Variable (y)")

    st.write(
        "Target variable: is_canceled"
    )

    st.dataframe(
        y.head(10),
        use_container_width=True
    )

    # -----------------------------------------
    # Shapes
    # -----------------------------------------

    st.subheader("4️⃣ Feature and Target Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Feature Matrix X",
            str(X.shape)
        )

    with col2:
        st.metric(
            "Target Vector y",
            str(y.shape)
        )

    # -----------------------------------------
    # Statistics
    # -----------------------------------------

    st.subheader("5️⃣ Feature Statistics")

    st.dataframe(
        X.describe().T,
        use_container_width=True
    )

    st.success(
        "✅ Feature representation completed successfully!"
    )


# -----------------------------------------
# Experiment 7 - Model Selection & Data Split
# -----------------------------------------

elif experiment == "Experiment 7 - Model Selection & Data Split":

    st.header("🤖 Experiment 7 - Model Selection & Data Split")

    st.write(
        "In this experiment, classification models are selected "
        "and the dataset is divided into training and testing data."
    )

    # Load and clean dataset
    cleaned_df = df.drop(columns=["company"]).dropna()

    # Remove target and leakage columns
    columns_to_remove = [
        "is_canceled",
        "reservation_status",
        "reservation_status_date"
    ]

    X_raw = cleaned_df.drop(
        columns=columns_to_remove
    )

    y = cleaned_df["is_canceled"]

    # Encode categorical columns
    categorical_columns = X_raw.select_dtypes(
        include="object"
    ).columns

    X = pd.get_dummies(
        X_raw,
        columns=categorical_columns,
        drop_first=True
    )

    X = X.astype(int)

    # -----------------------------------------
    # Classification Models
    # -----------------------------------------

    st.subheader("1️⃣ Classification Models")

    models = {
        "Logistic Regression":
            "Linear classification model",

        "Decision Tree":
            "Tree-based classification model",

        "Random Forest":
            "Ensemble classification model"
    }

    model_table = pd.DataFrame(
        models.items(),
        columns=["Model", "Description"]
    )

    st.dataframe(
        model_table,
        use_container_width=True
    )

    # -----------------------------------------
    # Select Model
    # -----------------------------------------

    st.subheader("2️⃣ Select Classification Model")

    selected_model = st.selectbox(
        "Choose a model for this project:",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ]
    )

    st.info(
        f"Selected Model: {selected_model}"
    )

    # Save selected model
    st.session_state["selected_model"] = selected_model

    # -----------------------------------------
    # Train-Test Split
    # -----------------------------------------

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    st.subheader("3️⃣ Train-Test Split")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Training Records",
            X_train.shape[0]
        )

    with col2:
        st.metric(
            "Testing Records",
            X_test.shape[0]
        )

    # -----------------------------------------
    # Feature Matrix Shape
    # -----------------------------------------

    st.subheader("4️⃣ Feature Matrix Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            "X Train:",
            X_train.shape
        )

        st.write(
            "y Train:",
            y_train.shape
        )

    with col2:
        st.write(
            "X Test:",
            X_test.shape
        )

        st.write(
            "y Test:",
            y_test.shape
        )

    # -----------------------------------------
    # Target Distribution
    # -----------------------------------------

    st.subheader("5️⃣ Target Distribution")

    target_table = y.value_counts().rename(
        index={
            0: "Not Cancelled",
            1: "Cancelled"
        }
    )

    st.bar_chart(target_table)

    st.success(
        "✅ Model selection and train-test split completed!"
    )


# -----------------------------------------
# Experiment 8 - Model Training
# -----------------------------------------

elif experiment == "Experiment 8 - Model Training":

    st.header("🧠 Experiment 8 - Model Training")

    st.write(
        "In this experiment, the selected classification "
        "model is trained using the training dataset."
    )

    # Load and clean dataset
    cleaned_df = df.drop(columns=["company"]).dropna()

    # Remove target and leakage columns
    columns_to_remove = [
        "is_canceled",
        "reservation_status",
        "reservation_status_date"
    ]

    X_raw = cleaned_df.drop(
        columns=columns_to_remove
    )

    y = cleaned_df["is_canceled"]

    # Encode categorical columns
    categorical_columns = X_raw.select_dtypes(
        include="object"
    ).columns

    X = pd.get_dummies(
        X_raw,
        columns=categorical_columns,
        drop_first=True
    )

    X = X.astype(int)

    # Train-test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -----------------------------------------
    # Get Selected Model
    # -----------------------------------------

    selected_model = st.session_state.get(
        "selected_model",
        "Random Forest"
    )

    st.subheader("1️⃣ Selected Model")

    st.info(
        f"Model selected for training: {selected_model}"
    )

    # -----------------------------------------
    # Import Models
    # -----------------------------------------

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    # -----------------------------------------
    # Create Model
    # -----------------------------------------

    if selected_model == "Logistic Regression":

        model = LogisticRegression(
            max_iter=1000
        )

    elif selected_model == "Decision Tree":

        model = DecisionTreeClassifier(
            random_state=42
        )

    else:

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    # -----------------------------------------
    # Train Model
    # -----------------------------------------

    st.subheader("2️⃣ Training the Model")

    with st.spinner(
        "Training the model... Please wait."
    ):

        model.fit(
            X_train,
            y_train
        )

    st.success(
        "✅ Model trained successfully!"
    )

    # Save trained model
    st.session_state["model"] = model
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["X_train"] = X_train
    st.session_state["y_train"] = y_train

    # -----------------------------------------
    # Training Information
    # -----------------------------------------

    st.subheader("3️⃣ Training Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Training Records",
            X_train.shape[0]
        )

    with col2:
        st.metric(
            "Testing Records",
            X_test.shape[0]
        )

    with col3:
        st.metric(
            "Number of Features",
            X_train.shape[1]
        )

    # -----------------------------------------
    # Model Details
    # -----------------------------------------

    st.subheader("4️⃣ Model Details")

    st.write(
        "Selected Model:",
        selected_model
    )

    st.write(
        "Model Status:",
        "Trained Successfully"
    )

    st.write(
        "The model has been fitted using the training data."
    )

    st.success(
        "🎉 Experiment 8 completed successfully!"
    )


# -----------------------------------------
# Experiment 9 - Prediction
# -----------------------------------------

elif experiment == "Experiment 9 - Prediction":

    st.header("🔮 Experiment 9 - Prediction")

    st.write(
        "This experiment demonstrates prediction using "
        "labelled test data and new unlabelled data."
    )

    # Check whether model exists
    if "model" not in st.session_state:

        st.warning(
            "⚠️ Please complete Experiment 8 - Model Training first."
        )

    else:

        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        # -----------------------------------------
        # Labelled Data Prediction
        # -----------------------------------------

        st.subheader(
            "1️⃣ Prediction Using Labelled Test Data"
        )

        test_sample = X_test.iloc[[0]]

        actual_value = y_test.iloc[0]

        predicted_value = model.predict(
            test_sample
        )[0]

        actual_label = (
            "Not Cancelled"
            if actual_value == 0
            else "Cancelled"
        )

        predicted_label = (
            "Not Cancelled"
            if predicted_value == 0
            else "Cancelled"
        )

        st.write(
            "Actual Label:",
            actual_label
        )

        st.write(
            "Predicted Label:",
            predicted_label
        )

        if actual_value == predicted_value:

            st.success(
                "✅ Prediction matches the actual label."
            )

        else:

            st.warning(
                "⚠️ Prediction does not match the actual label."
            )

        # -----------------------------------------
        # New Unlabelled Data
        # -----------------------------------------

        st.subheader(
            "2️⃣ Prediction Using New Unlabelled Data"
        )

        st.write(
            "A new booking record is selected without using "
            "its actual target label."
        )

        new_data = X_test.iloc[[1]]

        st.write(
            "New Unlabelled Input:"
        )

        st.dataframe(
            new_data,
            use_container_width=True
        )

        new_prediction = model.predict(
            new_data
        )[0]

        new_prediction_label = (
            "Not Cancelled"
            if new_prediction == 0
            else "Cancelled"
        )

        st.success(
            f"Predicted Result: {new_prediction_label}"
        )

        st.success(
            "🎉 Experiment 9 completed successfully!"
        )


# -----------------------------------------
# Experiment 10 - Performance Metrics
# -----------------------------------------

elif experiment == "Experiment 10 - Performance Metrics":

    st.header("📈 Experiment 10 - Performance Metrics")

    st.write(
        "This experiment evaluates the trained classification "
        "model using performance metrics and a confusion matrix."
    )

    # Check whether model exists
    if "model" not in st.session_state:

        st.warning(
            "⚠️ Please complete Experiment 8 - Model Training first."
        )

    else:

        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        # Make predictions
        y_pred = model.predict(
            X_test
        )

        # -----------------------------------------
        # Import Metrics
        # -----------------------------------------

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix
        )

        # -----------------------------------------
        # Performance Metrics
        # -----------------------------------------

        st.subheader("1️⃣ Performance Metrics")

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy:.2%}"
            )

            st.metric(
                "Precision",
                f"{precision:.2%}"
            )

        with col2:

            st.metric(
                "Recall",
                f"{recall:.2%}"
            )

            st.metric(
                "F1 Score",
                f"{f1:.2%}"
            )

        # -----------------------------------------
        # Confusion Matrix
        # -----------------------------------------

        st.subheader("2️⃣ Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Not Cancelled",
                "Actual Cancelled"
            ],
            columns=[
                "Predicted Not Cancelled",
                "Predicted Cancelled"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        st.success(
            "🎉 Experiment 10 completed successfully!"
        )
        
    # -----------------------------------------
# Download Cleaned Dataset
# -----------------------------------------

elif experiment == "📥 Download Cleaned Dataset":

    st.header("📥 Download Cleaned Dataset")

    st.write(
        "The dataset has been cleaned by removing the "
        "unwanted company column and missing values."
    )

    cleaned_df = df.drop(
        columns=["company"]
    ).dropna()

    st.subheader("📋 Cleaned Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Cleaned Records",
            cleaned_df.shape[0]
        )

    with col2:
        st.metric(
            "Cleaned Features",
            cleaned_df.shape[1]
        )

    st.dataframe(
        cleaned_df,
        use_container_width=True,
        height=500
    )

    csv_data = cleaned_df.to_csv(
        index=False
    )

    st.subheader("⬇️ Download")

    st.write(
        "Click the button below to download the "
        "complete cleaned dataset."
    )

    st.download_button(
        label="⬇️ Download Cleaned Dataset",
        data=csv_data,
        file_name="cleaned_hotel_bookings.csv",
        mime="text/csv"
    )

    st.success(
        "✅ Cleaned dataset is ready for download!"
    )