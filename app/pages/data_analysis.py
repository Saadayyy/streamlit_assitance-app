import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def app():
    st.title("Data Analysis")
    st.write("Analyze and visualize your dataset, with added fake data.")

    # Load and clean dataset
    dataset = load_and_clean_data()
    st.write("### Original Dataset (Cleaned)")
    st.dataframe(dataset.head())

    # Generate fake data
    fake_data = generate_fake_data(dataset, num_rows=int(len(dataset) * 0.25))
    fake_data = clean_fake_data(fake_data, dataset)
    st.write("### Fake Dataset")
    st.dataframe(fake_data.head())

    # Combine datasets
    combined_data = pd.concat([dataset, fake_data], ignore_index=True)
    combined_data = validate_combined_data(combined_data)
    st.write("### Combined Dataset (Cleaned)")

    # Paginate the dataset display
    show_paginated_data(combined_data)

    # Basic Statistics
    st.write("### Descriptive Statistics")
    display_statistics(combined_data)

    # Outlier Handling
    st.write("### Outlier Handling")
    outlier_checkbox = st.checkbox("Remove Outliers")
    if outlier_checkbox:
        combined_data = remove_outliers(combined_data)
        st.write("Outliers removed based on IQR method.")

    # Visualizations
    st.write("### Data Visualizations")
    plot_numerical_distributions(combined_data)

    # Data Transformation and Visualization
    st.write("### Data Transformation")
    transform_and_visualize(combined_data)


def load_and_clean_data():
    dataset_path = "data/dataset.csv"
    try:
        dataset = pd.read_csv(dataset_path, encoding='ISO-8859-1')
        dataset['Quantity'] = pd.to_numeric(dataset['Quantity'], errors='coerce')
        dataset.dropna(subset=['Quantity'], inplace=True)
        dataset['Quantity'] = dataset['Quantity'].astype(int)
        return dataset
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def generate_fake_data(reference_data, num_rows=100):
    fake = Faker()
    fake_data = pd.DataFrame()

    for column in reference_data.columns:
        if reference_data[column].dtype == np.number:
            fake_data[column] = np.random.uniform(
                reference_data[column].min(),
                reference_data[column].max(),
                num_rows,
            )
        elif pd.api.types.is_datetime64_any_dtype(reference_data[column]):
            fake_data[column] = [fake.date_between(start_date='-5y', end_date='today') for _ in range(num_rows)]
        else:
            fake_data[column] = [fake.word() for _ in range(num_rows)]

    return fake_data


def clean_fake_data(fake_data, reference_data):
    for column in reference_data.columns:
        if reference_data[column].dtype == np.number:
            fake_data[column] = pd.to_numeric(fake_data[column], errors='coerce')
        elif pd.api.types.is_datetime64_any_dtype(reference_data[column]):
            fake_data[column] = pd.to_datetime(fake_data[column], errors='coerce')
    fake_data.dropna(inplace=True)
    return fake_data


def validate_combined_data(data):
    if 'Quantity' in data.columns:
        data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
        data.dropna(subset=['Quantity'], inplace=True)
        data['Quantity'] = data['Quantity'].astype(int)
    return data


def show_paginated_data(data, page_size=100):
    """Display dataset in pages to handle large datasets."""
    total_rows = len(data)
    page_number = st.number_input("Page number", min_value=1, max_value=(total_rows // page_size) + 1, step=1)
    start_row = (page_number - 1) * page_size
    end_row = start_row + page_size
    st.dataframe(data.iloc[start_row:end_row])


def display_statistics(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        stats = data[numeric_cols].describe().transpose()
        st.dataframe(stats)


def remove_outliers(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        data = data[(data[col] >= Q1 - 1.5 * IQR) & (data[col] <= Q3 + 1.5 * IQR)]
    return data


def plot_numerical_distributions(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        st.warning("No numeric columns found for plotting.")
        return

    for col in numeric_cols:
        st.write(f"#### {col}")
        fig, ax = plt.subplots()
        data[col].plot(kind="box", ax=ax)
        st.pyplot(fig)

    st.write("### Correlation Matrix")
    try:
        corr = data[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting correlation matrix: {e}")


def transform_and_visualize(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if numeric_cols.empty:
        st.warning("No numeric columns found for transformation.")
        return

    scaler_choice = st.radio("Choose a transformation", ["None", "Min-Max Scaling", "Standardization"])
    if scaler_choice == "None":
        st.write("No transformation applied.")
        return

    scaler = MinMaxScaler() if scaler_choice == "Min-Max Scaling" else StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)

    for col in numeric_cols:
        st.write(f"#### {col} (Transformed)")
        fig, ax = plt.subplots()
        scaled_df[col].plot(kind="box", ax=ax)
        st.pyplot(fig)


if __name__ == "__main__":
    app()
