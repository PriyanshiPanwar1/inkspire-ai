import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from rag import ingest_docs, retrieve_context
from agents import run_workflow
from genai import generate_content

# ---- STYLE ----
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.stButton>button {
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
.stTextArea textarea {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---- SETUP ----
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "meta-llama/llama-3-8b-instruct"
# ---- LLM WRAPPER ----
def call_llm(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# ---- PAGE CONFIG ----
st.set_page_config(page_title="Inkspire AI", page_icon="✨", layout="wide")

# ---- HEADER ----
st.markdown("""
# ✨ Inkspire AI  
### From thought to content — instantly
""")

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Settings")

    tone = st.selectbox("Tone", ["formal", "casual", "persuasive"])
    content_type = st.selectbox("Content Type", ["blog", "social media", "script"])

    st.divider()

    st.subheader("📄 RAG Settings")
    use_rag = st.checkbox("Use RAG")
    show_steps = st.checkbox("Show Agent Steps", value=True)

    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

    if st.button("📥 Ingest Documents"):
        if uploaded_files:
            msg = ingest_docs(uploaded_files)
            st.success(msg)

    st.divider()

    st.subheader("✨ Content Enhancements")
    add_caption = st.checkbox("Add Caption")
    add_hashtags = st.checkbox("Add Trending Hashtags")
    add_hook = st.checkbox("Add Hook")
    add_keywords = st.checkbox("Add SEO Keywords")


# ---- MAIN AREA ----
st.subheader("🧠 Content Generator")

prompt = st.text_area(
    "Enter your topic",
    placeholder="e.g., AI in climate change",
    height=150
)

col1, col2 = st.columns(2)

with col1:
    simple_btn = st.button("✨ Generate (Simple)")

with col2:
    workflow_btn = st.button("🚀 Run Full Workflow")

st.divider()

# ---- OUTPUT ----
if simple_btn:
    if prompt:
        with st.spinner("Generating content..."):
            result = generate_content(
                prompt,
                tone,
                content_type,
                call_llm,
                use_rag,
                add_caption,
                add_hashtags,
                add_hook,
                add_keywords
            )
            st.markdown("## 📄 Output")
            st.markdown(result)
    else:
        st.warning("Please enter a topic")

if workflow_btn:
    if prompt:
        with st.spinner("Running AI workflow..."):
            result = run_workflow(
                prompt,
                tone,
                content_type,
                call_llm,
                use_rag,
                show_steps
            )
            st.markdown("## 🤖 AI Workflow Output")
            st.markdown(result)
    else:
        st.warning("Please enter a topic")