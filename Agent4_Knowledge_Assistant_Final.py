import os
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from bidi.algorithm import get_display
import pickle
import time

# ==========================================
# 1. הגדרות API ורשימת מודלים (Rotation)
# ==========================================
client = genai.Client(api_key=API_KEY)

BASE_PATH = r"C:\Users\lidor.kedmi\Desktop\Final_Project\V2"
MASTER_FOLDER = os.path.join(BASE_PATH, 'Master_Knowledge_Base')
INDEX_FILE = os.path.join(BASE_PATH, 'knowledge_index.pkl')

# שימוש בשם המודל המדויק שקיבלנו מהדיאגנוסטיקה
EMBEDDING_MODEL = "models/gemini-embedding-2"

MODELS_LIST = [
    'models/gemini-3-flash-preview',
    'models/gemini-2.5-flash',
    'models/gemini-2.5-flash-lite'
]

# ==========================================
# 2. פונקציות עזר: וקטוריזציה
# ==========================================
def get_embedding(text):
    """קבלת וקטור (Embedding) עבור טקסט מ-Gemini API"""
    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"❌ Error getting embedding with {EMBEDDING_MODEL}: {e}")
        return None

def build_index():
    """סריקת קבצי המאסטר ובניית אינדקס וקטורי עם ניהול מכסות (Retry Logic)"""
    if not os.path.exists(MASTER_FOLDER):
        print(f"❌ Error: Folder {MASTER_FOLDER} not found!")
        return []

    print("📂 Building knowledge index from Master files...")
    files = [f for f in os.listdir(MASTER_FOLDER) if f.endswith('.txt')]
    
    knowledge_data = []
    for f_name in files:
        success = False
        attempts = 0
        
        while not success and attempts < 3: # ננסה עד 3 פעמים לכל קובץ
            with open(os.path.join(MASTER_FOLDER, f_name), 'r', encoding='utf-8-sig') as f:
                content = f.read()
                vector = get_embedding(content)
                
                if vector is not None:
                    knowledge_data.append({
                        "file_name": f_name,
                        "content": content,
                        "vector": vector
                    })
                    print(f"✅ Indexed: {f_name}")
                    success = True
                else:
                    # אם קיבלנו None (שזה מה ש-get_embedding מחזירה בשגיאת 429)
                    attempts += 1
                    print(f"⚠️ Quota reached on {f_name}. Waiting 10 seconds (Attempt {attempts})...")
                    time.sleep(10) # המתנה משמעותית כדי "לשחרר" את החסימה בשרת
        
        # השהיה קבועה של שנייה בין קבצים כדי למנוע חסימה מראש
        time.sleep(1)
    
    if knowledge_data:
        with open(INDEX_FILE, 'wb') as f:
            pickle.dump(knowledge_data, f)
        print(f"💾 Index saved to {INDEX_FILE} with {len(knowledge_data)} files.")
    
    return knowledge_data
# ==========================================
# 3. מנוע השאילתות
# ==========================================
def ask_assistant(query, index):
    query_vector = get_embedding(user_query)
    if query_vector is None: return "שגיאה ביצירת וקטור השאילתה."

    all_vectors = np.array([item['vector'] for item in index])
    similarities = cosine_similarity([query_vector], all_vectors)[0]
    
    top_indices = np.argsort(similarities)[-3:][::-1]
    context_parts = []
    
    for idx in top_indices:
        if similarities[idx] > 0.2: # סף רלוונטיות
            context_parts.append(f"Source: {index[idx]['file_name']}\nContent: {index[idx]['content']}")

    if not context_parts:
        return "לא נמצא מידע רלוונטי במאגר הידע הארגוני."

    context_text = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""
    You are a professional Industrial Engineering Assistant. 
    Answer the user's question ONLY based on the provided Master KT documents.
    Always mention the specific source file name in your answer.
    Answer in Hebrew.

    CONTEXT:
    {context_text}

    QUESTION:
    {query}
    """

    for model_name in MODELS_LIST:
        try:
            print(f"    🔄 Generating answer with: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            if response and response.text:
                return response.text
        except Exception as e:
            continue
    
    return "שגיאה בקבלת מענה מהמודלים."

# ==========================================
# 4. ממשק הרצה
# ==========================================
if __name__ == "__main__":
    if os.path.exists(INDEX_FILE):
        print("💡 Loading existing index...")
        with open(INDEX_FILE, 'rb') as f:
            knowledge_index = pickle.load(f)
    else:
        knowledge_index = build_index()

    if not knowledge_index:
        print("❌ Knowledge index is empty. Check your Master files.")
    else:
        print("\n--- 🤖 IEM Knowledge Assistant Ready ---")
        while True:
            user_query = input(get_display("👨‍💼 שאלה: "))
            if user_query.lower() in ['exit', 'quit', 'צא']: break
            
            print("⏳ Searching...")
            answer = ask_assistant(user_query, knowledge_index)
            print(get_display(f"\n💡 תשובה:\n{answer}"))