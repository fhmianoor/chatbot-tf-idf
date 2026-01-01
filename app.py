import streamlit as st
import joblib
import string
import nltk
import time
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Chatbot TF-IDF",
    page_icon="",
    layout="centered"
)

nltk.download("punkt")
nltk.download("stopwords")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


if "page" not in st.session_state:
    st.session_state.page = 1

MESSAGES_PER_PAGE = 6


st.markdown("""
<h1 style="text-align:center;">Chatbot TF-IDF Bahasa Indonesia</h1>
<p style="text-align:center;color:gray;">
Chatbot berbasis cosine similarity & TF-IDF
</p>
""", unsafe_allow_html=True)


st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)


bg = "#0E1117" if st.session_state.dark_mode else "#FFFFFF"
user_bg = "#2E7D32" if st.session_state.dark_mode else "#DCF8C6"
bot_bg = "#1E1E1E" if st.session_state.dark_mode else "#F1F0F0"
text_color = "#FFFFFF" if st.session_state.dark_mode else "#000000"

st.markdown(f"""
<style>
html, body, [class*="css"] {{
    background-color: {bg};
    color: {text_color};
}}

.chat-container {{
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 20px;
}}

.user-bubble {{
    align-self: flex-end;
    background: {user_bg};
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
}}

.bot-bubble {{
    align-self: flex-start;
    background: {bot_bg};
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%;
}}

footer {{
    visibility: hidden;
}}
</style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "chatbot.pkl")

artifacts = joblib.load(MODEL_PATH)
vectorizer = artifacts["vectorizer"]
tfidf_matrix = artifacts["tfidf_matrix"]
df = artifacts["data"]

stop_words = set(stopwords.words("indonesian"))


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def get_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    idx = similarity.argmax()
    score = similarity[0][idx]

    if score < 0.2:
        return "Maaf, saya belum memahami pertanyaan tersebut."
    return df.iloc[idx]["answer"]


with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "",
        placeholder="Ketik pertanyaan lalu tekan Enter...",
        label_visibility="collapsed"
    )
    send = st.form_submit_button("Send")

if send and user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("sedang mengetik..."):
        time.sleep(0.6)
        bot_response = get_response(user_input)

    st.session_state.chat_history.append(("bot", bot_response))

    st.session_state.page = 1


total_messages = len(st.session_state.chat_history)

start_idx = max(
    0,
    total_messages - st.session_state.page * MESSAGES_PER_PAGE
)

visible_messages = st.session_state.chat_history[start_idx:]


if start_idx > 0:
    if st.button("Load older messages"):
        st.session_state.page += 1

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for role, msg in visible_messages:
    if role == "user":
        st.markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<hr>
<p style="text-align:center;font-size:12px;color:gray;">
TF-IDF Chatbot • Streamlit • NLP Indonesia
</p>
""", unsafe_allow_html=True)
