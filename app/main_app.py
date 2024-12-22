import streamlit as st

# Streamlit App Configuration
st.set_page_config(
    page_title="Data-Driven Web App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = {
    "Data Analysis": "pages.data_analysis",
    "Model Training": "pages.model_training",
    "Chatbot Integration": "pages.chatbot_integration",
}

selection = st.sidebar.radio("Go to", list(pages.keys()))

# Dynamically load selected page
if selection in pages:
    module = __import__(pages[selection], fromlist=[""])
    module.app()
