import streamlit as st
import time

# Title
st.set_page_config(page_title="Simple Chatbot", layout="centered")
st.title("🤖 Chatbox App")

# Session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Simple bot reply function (you can replace this with your model)
def get_bot_response(user_message):
    # Add actual model or API here
    return f"Echo: {user_message}"

# Chat input box
user_input = st.text_input("You:", key="input", placeholder="Type your message and press Enter...")

# On Enter
if user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.spinner("Bot is typing..."):
        time.sleep(1)  # Simulate processing
        bot_response = get_bot_response(user_input)
        st.session_state.chat_history.append(("Bot", bot_response))
    st.experimental_rerun()  # Refresh to clear input and show updated history

# Show chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 {sender}:** {message}")
    else:
        st.markdown(f"**🤖 {sender}:** {message}")
