import streamlit as st
import requests


def app():
    st.title("Chatbot Integration")
    st.write("Talk to the Rasa chatbot.")

    user_input = st.text_input("You: ", key="user_input")
    if st.button("Send"):
        try:
            # Replace localhost with the IP/host where your Rasa bot is running
            response = requests.post(
                "http://localhost:5005/webhooks/rest/webhook",
                json={"sender": "user", "message": user_input},
            )
            bot_response = response.json()
            for message in bot_response:
                st.write(f"Bot: {message.get('text', 'Sorry, no response available.')}")
        except Exception as e:
            st.error(f"Error communicating with chatbot: {e}")
