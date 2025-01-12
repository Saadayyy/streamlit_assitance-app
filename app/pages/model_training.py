import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def app():
    st.title("Flexible Model Training Guide")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        dataset = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(dataset.head())

        # Missing Data Handling Options
        if st.checkbox("Handle Missing Data"):
            st.write("Options for handling missing data will be shown here.")
            # Add options like filling or dropping missing data

        # Feature Selection
        feature_cols = st.multiselect(
            "Select Features", dataset.columns, default=dataset.columns[:-1]
        )
        target_col = st.selectbox("Select Target Column", dataset.columns)

        if feature_cols and target_col:
            X = dataset[feature_cols]
            y = dataset[target_col]

            # Correlation Analysis
            if st.checkbox("Show Correlation Matrix"):
                st.write("### Correlation Matrix")
                st.dataframe(X.corr())

            # Model Selection and Training
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

            # Model Training
            if st.button("Train Model"):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # Result Interpretation
                if model_type == "Regression":
                    st.write(f"R² Score: {r2_score(y_test, predictions):.2f}")
                    st.write(
                        f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}"
                    )
                else:
                    st.write(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

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


if __name__ == "__app__":
    app()
