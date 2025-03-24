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
    
    # More explicit prompting with clear instructions
    system_prompt = "You are a helpful AI assistant that provides accurate and informative answers to questions. Answer the following question in detail:"
    
    # Create a clear conversation format
    if use_history and len(st.session_state.messages) > 0:
        # Build context from recent messages (limit to prevent context overflow)
        conversation = system_prompt + "\n\n"
        for msg in st.session_state.messages[-4:]:
            prefix = "Human: " if msg["role"] == "user" else "Assistant: "
            conversation += f"{prefix}{msg['content']}\n"
        
        # Add current question
        conversation += f"Human: {user_input}\nAssistant:"
    else:
        conversation = f"{system_prompt}\n\nHuman: {user_input}\nAssistant:"
    
    # Encode the conversation
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(conversation, return_tensors="pt").to(device)
    
    # Generate with more controlled parameters
    attention_mask = torch.ones(input_ids.shape, device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_length,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Stop generation at "Human:" to prevent the model from continuing the conversation
            bad_words_ids=[[tokenizer.encode("Human:", add_special_tokens=False)[0]]]
        )
    
    # Decode the generated response
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the assistant's part of the response
    response_parts = full_response.split("Assistant:")
    if len(response_parts) > 1:
        # Get the last part after "Assistant:"
        assistant_response = response_parts[-1].strip()
        
        # Clean up any parts that might contain "Human:" (end of response)
        if "Human:" in assistant_response:
            assistant_response = assistant_response.split("Human:")[0].strip()
    else:
        # Fallback if the format isn't as expected
        assistant_response = full_response.replace(conversation, "").strip()
    
    # Additional cleanup
    assistant_response = assistant_response.replace("<|endoftext|>", "").strip()
    
    return assistant_response
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
with st.form(key="message_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", key="user_input")
    submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        

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