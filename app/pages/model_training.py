import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder


def preprocess_data(dataset, categorical_cols, encoding_type):
    """
    Preprocess the dataset by encoding categorical columns.
    """
    if encoding_type == "Label Encoding":
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col].astype(str))
            label_encoders[col] = le
    elif encoding_type == "One-Hot Encoding":
        dataset = pd.get_dummies(dataset, columns=categorical_cols)
    return dataset


def app():
    # Page title
    st.title("Flexible Model Training Guide")

    # Instructions for users
    st.write("""
    ## Instructions
    Welcome to the Flexible Model Training Guide! This tool allows you to upload a dataset and quickly train 
    a regression or classification model. Here’s how to use it:
    
    1. **Upload Your Dataset**:
       - The file should be in CSV format.
       - Ensure your dataset includes both features (input data) and a target column (output data to predict).
       
    2. **Handle Missing Data (Optional)**:
       - If your dataset has missing values, consider handling them before training the model.

    3. **Select Features and Target Column**:
       - Choose the columns to use as features (inputs) and the column to predict (target).
       - For example, if you're predicting house prices, select columns like `size`, `location`, and `bedrooms` as features and `price` as the target.

    4. **Handle Categorical Data**:
       - If your dataset includes non-numeric data (e.g., days of the week or categories), choose how to encode it: 
         - **Label Encoding**: Assigns a unique integer to each category.
         - **One-Hot Encoding**: Creates a binary column for each category.

    5. **Train a Model**:
       - Choose a model type (regression or classification) and a specific model (e.g., linear regression or random forest).
       - Split the dataset into training and testing sets, then train the model.

    6. **View Results**:
       - For regression models, view metrics like R² and RMSE.
       - For classification models, view metrics like accuracy.
       - If the model supports it, you’ll also see feature importance to understand which features were most impactful.

    Start by uploading your dataset below!
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        dataset = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(dataset.head())

        # Identify categorical columns
        categorical_cols = dataset.select_dtypes(include=["object"]).columns.tolist()
        st.write(f"Categorical Columns: {categorical_cols}")

        # Encode categorical data if necessary
        if categorical_cols:
            encoding_choice = st.radio(
                "Choose Encoding Type for Categorical Features",
                ["Label Encoding", "One-Hot Encoding"],
            )
            dataset = preprocess_data(dataset, categorical_cols, encoding_choice)

        st.write("### Processed Dataset Preview")
        st.dataframe(dataset.head())

        # Feature selection
        feature_cols = st.multiselect(
            "Select Features", dataset.columns, default=dataset.columns[:-1]
        )
        target_col = st.selectbox("Select Target Column", dataset.columns)

        if feature_cols and target_col:
            X = dataset[feature_cols]
            y = dataset[target_col]

            # Ensure y is numeric (if it's a categorical target for classification)
            if y.dtype == "object" or y.dtype.name == "category":
                y = LabelEncoder().fit_transform(y)

            # Show correlation matrix
            if st.checkbox("Show Correlation Matrix"):
                st.write("### Correlation Matrix")
                st.dataframe(X.corr())

            # Model selection
            st.sidebar.header("Model Selection")
            model_type = st.sidebar.radio(
                "Choose Model Type", ["Regression", "Classification"]
            )

            if model_type == "Regression":
                model_choice = st.sidebar.selectbox(
                    "Select Model", ["Linear Regression", "Random Forest Regressor"]
                )
                model = (
                    LinearRegression()
                    if model_choice == "Linear Regression"
                    else RandomForestRegressor()
                )

            else:  # Classification
                model_choice = st.sidebar.selectbox(
                    "Select Model", ["Logistic Regression", "Random Forest Classifier"]
                )
                model = (
                    LogisticRegression()
                    if model_choice == "Logistic Regression"
                    else RandomForestClassifier()
                )

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Model training
            if st.button("Train Model"):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Result interpretation
                if model_type == "Regression":
                    st.write(f"R² Score: {r2_score(y_test, predictions):.2f}")
                    st.write(
                        f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}"
                    )
                else:
                    st.write(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

                # Feature importance (if applicable)
                if hasattr(model, "feature_importances_"):
                    st.write("### Feature Importance")
                    importance_df = pd.DataFrame(
                        {
                            "Feature": feature_cols,
                            "Importance": model.feature_importances_,
                        }
                    )
                    st.dataframe(
                        importance_df.sort_values(by="Importance", ascending=False)
                    )
    else:
        st.info("Please upload a CSV file to start.")


if __name__ == "__main__":
    app()
