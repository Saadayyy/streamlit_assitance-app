import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  # For caching preprocessed data


# Caching preprocessed data to avoid recalculating every time the app runs
@st.cache  # This decorator caches the function output and makes it faster on repeated runs
def load_and_preprocess_data():
    dataset_path = "data/dataset.csv"
    try:
        dataset = pd.read_csv(dataset_path, encoding="ISO-8859-1")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

    processed_data, scaler, encoder = preprocess_data(dataset)
    return processed_data, scaler, encoder


# Optimized preprocessing function
def preprocess_data(dataset):
    data = dataset.copy()

    # Handle missing values
    data["CustomerID"] = data["CustomerID"].fillna(
        -1
    )  # Fill missing CustomerID with -1
    data["Description"] = data["Description"].fillna("Unknown")

    # Convert InvoiceDate to datetime and extract features
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data = data.assign(
        InvoiceYear=data["InvoiceDate"].dt.year,
        InvoiceMonth=data["InvoiceDate"].dt.month,
        InvoiceDay=data["InvoiceDate"].dt.day,
        InvoiceHour=data["InvoiceDate"].dt.hour,
    )

    # Compute TotalCost
    data["TotalCost"] = data["Quantity"] * data["UnitPrice"]

    # Drop unnecessary columns
    data = data.drop(columns=["InvoiceNo", "StockCode", "Description", "InvoiceDate"])

    # Encode categorical variables
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    encoder = TargetEncoder(cols=categorical_cols)
    data = encoder.fit_transform(
        data, data["TotalCost"]
    )  # Use 'TotalCost' for encoding

    # Scale numeric columns
    scaler = MinMaxScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data, scaler, encoder


def app():
    st.title("Model Training")

    # Load and preprocess dataset
    processed_data, scaler, encoder = load_and_preprocess_data()
    if processed_data is None:
        return

    st.write("### Dataset Preview")
    st.dataframe(processed_data.head())

    # Feature and Target Selection
    st.sidebar.write("## Feature Selection")
    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    target_col = st.sidebar.selectbox("Select Target Column (to predict)", numeric_cols)
    feature_cols = [col for col in numeric_cols if col != target_col]

    correlation_matrix = processed_data.corr()
    st.write("### Correlation Analysis")
    st.write("Correlation with Target:")
    st.dataframe(correlation_matrix[target_col].sort_values(ascending=False))

    # Avoid multicollinearity by excluding highly correlated features
    threshold = 0.8
    highly_correlated = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                highly_correlated.add(colname)

    filtered_features = [f for f in feature_cols if f not in highly_correlated]
    selected_features = st.sidebar.multiselect(
        "Select Features", filtered_features, default=filtered_features
    )

    if not selected_features or not target_col:
        st.warning("Please select features and a target column.")
        return

    # Train-test split
    X = processed_data[selected_features]
    y = processed_data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model Selection
    st.sidebar.write("## Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select Model", ["Linear Regression", "Random Forest Regressor"]
    )

    if st.button("Train Model"):
        st.write("### Model Training Results")

        if model_choice == "Linear Regression":
            st.write("#### Linear Regression")
            try:
                lin_reg = LinearRegression()
                lin_reg.fit(X_train, y_train)
                lin_reg_predictions = lin_reg.predict(X_test)
                lin_reg_r2 = r2_score(y_test, lin_reg_predictions)
                lin_reg_rmse = np.sqrt(mean_squared_error(y_test, lin_reg_predictions))
                st.write(f"R² Score: {lin_reg_r2:.2f}")
                st.write(f"RMSE: {lin_reg_rmse:.2f}")
            except Exception as e:
                st.error(f"Linear Regression failed: {e}")

        elif model_choice == "Random Forest Regressor":
            st.write("#### Random Forest Regressor")
            try:
                rf = RandomForestRegressor(random_state=42)

                # Hyperparameter tuning using GridSearchCV
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_features": ["auto", "sqrt", "log2"],
                    "max_depth": [None, 10, 20, 30],
                }
                grid_search = GridSearchCV(
                    estimator=rf, param_grid=param_grid, cv=5, scoring="r2"
                )
                grid_search.fit(X_train, y_train)

                best_rf = grid_search.best_estimator_
                rf_predictions = best_rf.predict(X_test)
                rf_r2 = r2_score(y_test, rf_predictions)
                rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

                st.write(f"Best Parameters: {grid_search.best_params_}")
                st.write(f"R² Score: {rf_r2:.2f}")
                st.write(f"RMSE: {rf_rmse:.2f}")

                # Feature importance
                feature_importances = pd.DataFrame(
                    {
                        "Feature": selected_features,
                        "Importance": best_rf.feature_importances_,
                    }
                ).sort_values(by="Importance", ascending=False)
                st.write("### Feature Importance")
                st.dataframe(feature_importances)
            except Exception as e:
                st.error(f"Random Forest Regressor failed: {e}")


if __name__ == "__main__":
    app()
