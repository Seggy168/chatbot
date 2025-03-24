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
        index=2
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
    
    # Add a BOS token if the tokenizer doesn't add it automatically
    if not tokenizer.bos_token:
        tokenizer.add_special_tokens({'bos_token': '<|startoftext|>'})
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    
    # Simpler, more direct prompt that works better with GPT-2
    if use_history and len(st.session_state.messages) > 2:  # Only use history if we have a meaningful conversation
        # Format: Q: question A: answer Q: question A:
        conversation = ""
        for msg in st.session_state.messages[-4:]:  # Last 4 messages at most
            prefix = "Q: " if msg["role"] == "user" else "A: "
            conversation += f"{prefix}{msg['content']}\n\n"
        
        prompt = f"{conversation}Q: {user_input}\n\nA:"
    else:
        # Simple Q&A format works better for one-off questions with GPT-2
        prompt = f"Q: {user_input}\n\nA:"
    
    # Encode prompt
    device = next(model.parameters()).device
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate response with carefully chosen parameters
    # Lower temperature for more focused responses
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_length,
            temperature=0.6,  # Lower temperature for more factual responses
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            # Stop at "Q:" to prevent model continuing with new questions
            bad_words_ids=[[tokenizer.encode("Q:", add_special_tokens=False)[0]]]
        )
    
    # Decode response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the answer part
    response = full_output.split("A:")[-1].strip()
    
    # Clean up the response
    response = clean_response(response)
    
    return response

def clean_response(text):
    """Clean up the model's response to remove common artifacts."""
    # Remove any JSON-like structures
    if "{" in text and "}" in text:
        text = text.split("{")[0].strip()
    
    # Remove numbered lists that aren't intentional
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Skip lines that are just numbers or common GPT-2 artifacts
        if line.strip().isdigit() or line.strip() in ["<", "<<", ">", ">>", "..."]:
            continue
        cleaned_lines.append(line)
    
    text = "\n".join(cleaned_lines)
    
    # Remove any remaining special tokens
    text = text.replace("<|endoftext|>", "").replace("<|startoftext|>", "")
    
    # If the response is too short or seems nonsensical
    if len(text.split()) < 3:
        text = "I don't have enough information to answer that question properly."
    
    return text.strip()
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
# Chat input - fixed form handling
with st.form(key="message_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", key="user_input")
    submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        # Add user message to chat history ONCE
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
        
# Reset button
if st.sidebar.button("Reset Conversation"):
    st.session_state.messages = []
    st.session_state.conversation_history = ""
    try:
        st.experimental_rerun()
    except AttributeError:
        st.warning("App rerun functionality is not available. Please reload the page manually.")


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