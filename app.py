import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Setup ---
st.set_page_config(page_title="MediBot", layout="wide")
st.title("🩺 MediBot - Your Medical Assistant")

# --- Load Data ---
@st.cache_resource
def load_index():
    return faiss.read_index("new_faiss_index.idx")

@st.cache_resource
def load_text_chunks():
    with open("new_chunk_texts.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_llm():
    model_name = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1), tokenizer

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

index = load_index()
chunk_texts = load_text_chunks()
embedder = load_embedder()
llm, tokenizer = load_llm()
summarizer = load_summarizer()

# --- Helper Functions ---
def get_top_k_chunks(query, k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    results = [chunk_texts[i] for i in indices[0]]
    return results, distances[0]

def build_prompt(query, context_chunks, history, max_context_tokens=800):
    context = "\n\n".join(context_chunks)
    context_tokens = tokenizer.encode(context, truncation=True, max_length=max_context_tokens, return_tensors="pt")
    truncated_context = tokenizer.decode(context_tokens[0], skip_special_tokens=True)

    history_str = ""
    for user_q, bot_a in history[-3:]:
        history_str += f"User: {user_q}\nMediBot: {bot_a}\n"

    prompt = f"""You are a helpful medical assistant. Use the context and conversation history to answer the user's question.

Context:
{truncated_context}

Conversation History:
{history_str}

User: {query}
MediBot:"""
    return prompt

def summarize_answer(answer):
    try:
        summary = summarizer(answer, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
        return summary
    except:
        return "No summary available."

def suggest_follow_ups(query):
    if "treatment" in query.lower():
        return ["What are the side effects?", "Are there alternative therapies?"]
    elif "symptom" in query.lower():
        return ["When should I see a doctor?", "How can symptoms be managed?"]
    elif "diagnosis" in query.lower():
        return ["What tests are usually done?", "How accurate are these tests?"]
    else:
        return ["What are treatment options?", "How is this condition diagnosed?"]

def answer_question(query, history):
    top_chunks, distances = get_top_k_chunks(query)
    prompt = build_prompt(query, top_chunks, history)

    result = llm(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )[0]["generated_text"]
    answer = result.split("MediBot:")[-1].strip()
    summary = summarize_answer(answer)

    top_distance = distances[0]
    if top_distance < 0.5:
        confidence = "High"
    elif top_distance < 1.0:
        confidence = "Medium"
    else:
        confidence = "Low"

    suggestions = suggest_follow_ups(query)
    return answer, summary, confidence, suggestions

# --- Conversation Memory ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- User Input ---
query = st.text_input("💬 Ask a medical question (type 'exit' or 'new topic: ...' to reset):", key="query_input")

if query:
    if query.lower() in ["exit", "quit"]:
        st.session_state.history.clear()
        st.success("Session ended. Take care! 👋")
    elif query.lower().startswith("new topic:"):
        st.session_state.history.clear()
        query = query[len("new topic:"):].strip()
        st.info("🔄 Starting a new topic...")

    with st.spinner("🤖 MediBot is thinking..."):
        answer, summary, confidence, suggestions = answer_question(query, st.session_state.history)
        st.session_state.history.append((query, answer))

        st.markdown("### 🧠 MediBot's Answer")
        st.write(answer)

        st.markdown("### 🩺 Key Takeaway")
        st.success(summary)

        st.markdown(f"### 🔍 Confidence: `{confidence}`")

        if suggestions:
            st.markdown("### 🤔 You can also ask:")
            for s in suggestions:
                st.markdown(f"- {s}")

# --- History ---
if st.session_state.history:
    with st.expander("🕒 Conversation History"):
        for q, a in st.session_state.history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**MediBot:** {a}")
            st.markdown("---")