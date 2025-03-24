# Streamlit Web Interface for GPT Chatbot
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# Set page title and favicon
st.set_page_config(
    page_title="GPT Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for better appearance
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: row;
    align-items: flex-start;
}
.chat-message.user {
    background-color: #f0f2f6;
}
.chat-message.bot {
    background-color: #e3f2fd;
}
.chat-message .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 1rem;
}
.chat-message .message {
    flex-grow: 1;
}
.stTextInput > div > div > input {
    caret-color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

# Header
st.title("GPT Chatbot")
st.subheader("Ask me anything and I'll respond using GPT")

# Sidebar for model configuration
with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox(
        "Select Model", 
        ["gpt2", "gpt2-medium", "gpt2-large"], 
        index=0
    )
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    max_length = st.slider("Max Response Length", min_value=50, max_value=500, value=150, step=50)
    use_gpu = st.checkbox("Use GPU (if available)", value=False)

# Function to load the model
@st.cache_resource
def load_model(model_name, use_gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    return model, tokenizer, device

# Load model button
if st.sidebar.button("Load/Reload Model"):
    with st.spinner(f"Loading {model_name} model..."):
        st.session_state.model, st.session_state.tokenizer, device = load_model(model_name, use_gpu)
    st.sidebar.success(f"Model loaded successfully on {device}!")

# If model not loaded yet, load the default
if st.session_state.model is None:
    with st.spinner(f"Loading default model..."):
        st.session_state.model, st.session_state.tokenizer, device = load_model("gpt2", False)
    st.sidebar.success(f"Default model loaded on {device}!")

# Function to generate response
def generate_response(user_input, use_history=True):
    # Get model and tokenizer from session state
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    # Get conversation history
    conversation_history = st.session_state.conversation_history
    
    # Define special tokens
    bot_token = "<|bot|>:"
    user_token = "<|user|>:"
    
    # Prepare prompt with history if needed
    if use_history and conversation_history:
        prompt = f"{conversation_history}\n{user_token} {user_input}\n{bot_token}"
    else:
        prompt = f"{user_token} {user_input}\n{bot_token}"
    
    # Encode prompt
    device = next(model.parameters()).device
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    # Decode response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the bot's response
    response = generated_text[len(prompt):].strip()
    
    # Update conversation history
    if use_history:
        st.session_state.conversation_history = f"{prompt} {response}"
        
        # Limit conversation history
        if len(st.session_state.conversation_history.split()) > 1024:
            history_parts = st.session_state.conversation_history.split(f"\n{user_token}")
            st.session_state.conversation_history = f"\n{user_token}".join(history_parts[-3:])
    
    return response

# Display chat messages
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message['role']}">
            <div class="avatar">{avatar}</div>
            <div class="message">{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)

# Chat input
user_input = st.text_input("Your message:", key="user_input")

# Reset button
if st.sidebar.button("Reset Conversation"):
    st.session_state.messages = []
    st.session_state.conversation_history = ""
    st.experimental_rerun()

# When user submits a message
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.container():
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">🧑‍💻</div>
            <div class="message">{user_input}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show typing animation
    with st.container():
        typing_placeholder = st.empty()
        typing_placeholder.markdown(f"""
        <div class="chat-message bot">
            <div class="avatar">🤖</div>
            <div class="message">Thinking...</div>
        </div>
        """, unsafe_allow_html=True)
    
        # Generate and display bot response
        bot_response = generate_response(user_input)
        
        # Replace typing animation with response
        typing_placeholder.markdown(f"""
        <div class="chat-message bot">
            <div class="avatar">🤖</div>
            <div class="message">{bot_response}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    
    # Clear the input box
    st.session_state.user_input = ""