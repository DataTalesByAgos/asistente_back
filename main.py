import os
import re
import base64
import requests
import mimetypes
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from io import BytesIO
from gtts import gTTS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# cfg
load_dotenv()

app = FastAPI(title="Chat + Whisper STT + gTTS + PDF", version="4.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN") or "hf_tu_token_aqui"
MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"


# ---------------- UTILIDADES ----------------
def clean_ai_text(text: str) -> str:
    """Limpia trazas internas y formato markdown."""
    text = re.sub(r"<\s*think[^>]*>.*?<\s*/\s*think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[*_#>`~]", "", text)
    return text.strip()


def synthesize_audio(text: str, lang="es") -> str:
    """Convierte texto a voz (base64) usando gTTS."""
    if not text:
        return ""
    text = text.strip()
    tts = gTTS(text=text[:500], lang=lang)
    buffer = BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def extract_context(user_text: str, pdf_db, k: int = 3) -> str:
    """Busca contexto en la base semántica, compatible con cualquier versión."""
    if not pdf_db:
        return ""
    try:
        results = pdf_db.similarity_search(user_text, k=k)
        context_parts = []
        for r in results:
            if hasattr(r, "page_content"):
                context_parts.append(r.page_content)
            elif hasattr(r, "document") and hasattr(r.document, "page_content"):
                context_parts.append(r.document.page_content)
        return "\n".join(context_parts)
    except Exception as e:
        print(f"⚠️ Error extrayendo contexto: {e}")
        return ""


def detect_intent(user_text: str, pdf_db) -> str:
    """Clasifica la intención del usuario."""
    text_low = user_text.lower()

    # saludos o sociales
    if any(p in text_low for p in [
        "hola", "buen día", "buenas", "cómo estás", "gracias", "chau", "adiós",
        "se escucha", "me oís", "me oyes", "te escucho", "probando", "que tal"
    ]):
        return "social"

    # si hay PDF y encuentra contexto relevante
    context = extract_context(user_text, pdf_db, k=1)
    if context.strip():
        return "contextual"

    return "desconocido"


def generate_reply(user_text: str, intent: str, pdf_db) -> str:
    """Genera respuesta según intención."""
    if intent == "social":
        return "¡Hola! Sí, te escucho perfectamente. ¿En qué puedo ayudarte?"

    if intent == "desconocido":
        return "No tengo esa información en la base de datos."

    # Si es contextual:
    context = extract_context(user_text, pdf_db, k=3)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente útil que responde exclusivamente en español. "
                    "Nunca muestres razonamientos, explicaciones internas ni texto en inglés. "
                    "Responde de forma clara, natural y concisa, basándote únicamente en la información del contexto. "
                    "Si el contexto no contiene la respuesta, responde exactamente: "
                    "'No tengo esa información en la base de datos.'"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Contexto:\n{context}\n\n"
                    f"Pregunta del usuario: {user_text}\n\n"
                    "Respuesta en español:"
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }

    try:
        resp = requests.post(
            HF_CHAT_URL,
            headers={**HF_HEADERS, "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        data = resp.json()
        ai_reply = clean_ai_text(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
        return ai_reply or "No tengo esa información en la base de datos."
    except Exception as e:
        print(f"Error conectando al modelo: {e}")
        return "Hubo un error al generar la respuesta."


# Carga vector DB o la crea desde db.pdf
def load_or_create_vector_db(pdf_path="db.pdf", db_dir="vector_db"):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(db_dir):
        print("Cargando base semántica existente...")
        return FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(pdf_path):
        print("No se encontró db.pdf, continuando sin base.")
        return None

    print("Creando nueva base semántica desde db.pdf...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(db_dir)
    print("Base FAISS creada y guardada.")
    return db


pdf_db = load_or_create_vector_db()

# Endpoint chat texto
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_text = data.get("text", "").strip()
    except Exception:
        form = await request.form()
        user_text = form.get("text", "").strip()

    if not user_text:
        return {"error": "Texto vacío."}

    print(f"🗣️ Usuario: {user_text}")

    intent = detect_intent(user_text, pdf_db)
    print(f"Intención detectada: {intent}")

    reply = generate_reply(user_text, intent, pdf_db)
    print(f"Respuesta: {reply}")

    return {"text": reply, "audio_base64": synthesize_audio(reply)}


# Endpoint chat audio
@app.post("/chat/audio")
async def chat_audio(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"Archivo recibido: {file.filename} → {tmp_path}")

        mime_type = mimetypes.guess_type(tmp_path)[0] or "audio/wav"
        with open(tmp_path, "rb") as f:
            audio_data = f.read()

        whisper_resp = requests.post(
            WHISPER_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Accept": "application/json",
                "Content-Type": mime_type,
            },
            data=audio_data,
            timeout=120,
        )
        whisper_json = whisper_resp.json()
        user_text = (whisper_json.get("text") or whisper_json.get("transcription", "")).strip()

        if not user_text:
            return {"error": "No se obtuvo transcripción del audio."}

        print(f"🗣️ Transcripción Whisper: {user_text}")

        intent = detect_intent(user_text, pdf_db)
        print(f"Intención detectada: {intent}")

        reply = generate_reply(user_text, intent, pdf_db)
        print("Conversación completada correctamente.")

        return {
            "input_text": user_text,
            "reply_text": reply,
            "audio_base64": synthesize_audio(reply)
        }

    except Exception as e:
        print(f"❌ Error procesando audio: {e}")
        return {"error": str(e)}


# Ruta raíz
@app.get("/")
def root():
    return {"message": "API Whisper + Chat + gTTS + PDF funcionando correctamente"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
