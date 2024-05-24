import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# streamlit cache clear (or menu)
@st.cache_resource
def load_data():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

st.set_page_config(page_title="Chatbot")

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        margin-top: 10px;
    }
    .stTextInput input {
        border: 2px solid #ff4b4b;
        border-radius: 10px;
        padding: 10px;
        width: 100%;
        font-size: 16px;
    }
    .stTextInput label {
        font-weight: bold;
        font-size: 18px;
    }
    .chat-container {
        padding: 10px;
        border: 2px solid #ff4b4b;
        border-radius: 10px;
        margin-top: 10px;
        background-color: #f5f5f5;
    }
    .chat-bubble {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        font-size: 16px;
    }
    .chat-bubble.user {
        background-color: #ff6666;
        text-align: left;
        color: white;
    }
    .chat-bubble.bot {
        background-color: #6699ff;
        text-align: left;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.write("# Welcome to the DialoGPT-medium Chatbot by GALLOMOR")
tokenizer, model = load_data()

# Init Session State
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.chat_history_ids = None
    st.session_state.history = []
else:
    st.session_state.step += 1

input = st.text_input(label="Enter your question:")

if input:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last output tokens from bot
    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.session_state.history.append(("User", input))
    st.session_state.history.append(("Bot", response))

if st.button('Reset'):
    st.session_state.step = 0
    st.session_state.chat_history_ids = None
    st.session_state.history = []

st.write("## Chat History")
chat_container = st.container()
with chat_container:
    for person, message in st.session_state.history:
        css_class = "user" if person == "User" else "bot"
        st.markdown(f'<div class="chat-bubble {css_class}"><strong>{person}:</strong> {message}</div>', unsafe_allow_html=True)

st.write("### Debug Info")
st.write(f"Step: {st.session_state.step}")
st.write(st.session_state.chat_history_ids)
