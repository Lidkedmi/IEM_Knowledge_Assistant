import streamlit as st
import os
import pickle
import numpy as np
from google import genai
from bidi.algorithm import get_display
from sklearn.metrics.pairwise import cosine_similarity

# פונקציית עזר לעברית
def heb(text):
    return get_display(str(text))

# הגדרות נתיבים יחסיים (מתאים לענן)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_PATH, 'knowledge_index.pkl')

# הגדרת כותרת האפליקציה
st.set_page_config(page_title="IEM Knowledge Assistant", page_icon="🤖")
st.title(heb("🤖 עוזר הידע של IEM"))
st.markdown(heb("ברוכים הבאים למערכת ניהול הידע מבוססת סוכנים."))

# טעינת מפתח ה-API בצורה מאובטחת מהגדרות הענן
if "GEMINI_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error(heb("מפתח API לא נמצא בהגדרות המערכת!"))
    st.stop()

@st.cache_resource
def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'rb') as f:
            return pickle.load(f)
    return None

knowledge_index = load_index()

def ask_assistant(query, index):
    # יצירת Embedding לשאילתה
    res = client.models.embed_content(model="models/gemini-embedding-2", contents=query)
    query_vector = np.array(res.embeddings[0].values).reshape(1, -1)
    
    # חישוב דמיון סמנטי
    doc_vectors = np.array([item['vector'] for item in index])
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    
    # שליפת 3 המסמכים הרלוונטיים ביותר (RAG)
    top_indices = np.argsort(similarities)[-3:][::-1]
    context = ""
    sources = []
    for idx in top_indices:
        context += f"\n---\nSource: {index[idx]['file_name']}\n{index[idx]['content']}\n"
        sources.append(index[idx]['file_name'])
    
    # בניית הפרומפט הסופי למודל
    prompt = f"""
    You are a professional industrial engineering assistant.
    Answer the user's question ONLY based on the context provided below.
    If the answer is not in the context, say that you don't know based on the organizational knowledge.
    
    Context:
    {context}
    
    Question: {query}
    """
    
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text, sources

# ממשק הצ'אט
if knowledge_index:
    user_query = st.text_input(heb("מה תרצה לדעת?"), placeholder=heb("הקלד את שאלתך כאן..."))
    
    if user_query:
        with st.spinner(heb("סורק את מאגר הידע...")):
            try:
                answer, sources = ask_assistant(user_query, knowledge_index)
                st.subheader(heb("💡 תשובה:"))
                st.write(heb(answer))
                st.info(heb(f"מקורות: {', '.join(sources)}"))
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.warning(heb("קובץ האינדקס לא נמצא. וודא שהעלית את knowledge_index.pkl ל-GitHub."))