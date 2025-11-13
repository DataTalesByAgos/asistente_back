import os
import re
import base64
import requests
import mimetypes
import json
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from io import BytesIO
from gtts import gTTS

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

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
HF_EMBEDDINGS_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"

USE_REMOTE_EMBEDDINGS = os.getenv("USE_REMOTE_EMBEDDINGS", "0") in ["1", "true", "True"]


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
    # Ensure DB is loaded lazily to reduce memory at startup
    pdf_db = ensure_pdf_db_loaded()
    if not pdf_db:
        return ""
    try:
        # Support both FAISS-style stores and our remote JSON index
        if isinstance(pdf_db, dict) and pdf_db.get("type") == "remote":
            results = similarity_search_remote(user_text, k=k, index_path=pdf_db.get("index_path", "vectorstore/remote_index.json"))
            context_parts = [r.get("page_content", "") for r in results]
            return "\n".join(context_parts)

        # Otherwise assume it's a FAISS-like object with similarity_search
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


pdf_db = None


def compute_embedding_remote(text: str) -> list:
    """Compute embedding via Hugging Face Inference API (remote).

    Returns a list of floats or raises Exception on error.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN no configurado para embeddings remotos")

    try:
        resp = requests.post(
            HF_EMBEDDINGS_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json={"inputs": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Possible shapes:
        # - flat list: [0.1, 0.2, ...]
        # - list of lists: [[...], [...], ...] (token embeddings) -> mean-pool
        # - dict or list of dicts with 'embedding' key

        if isinstance(data, dict) and "embedding" in data:
            return data["embedding"]

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "embedding" in data[0]:
            return data[0]["embedding"]

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
            return data

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # mean-pool token vectors
            dim = len(data[0])
            sums = [0.0] * dim
            count = 0
            for vec in data:
                if not isinstance(vec, list) or len(vec) != dim:
                    continue
                for i, v in enumerate(vec):
                    sums[i] += float(v)
                count += 1
            if count == 0:
                raise RuntimeError("Embeddings vacíos o formato inconsistente")
            return [s / count for s in sums]

        raise RuntimeError(f"Formato inesperado de embeddings: {data}")
    except Exception as e:
        print(f"Error obteniendo embedding remoto: {e}")
        raise


def build_remote_index_from_pdf(pdf_path="db.pdf", index_path="vectorstore/remote_index.json"):
    """Build a simple JSON-backed index using remote embeddings.

    Splits the PDF into chunks and stores embeddings for each chunk in a JSON file.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
    except Exception as e:
        print(f"No se pudieron importar loaders/text-splitters: {e}")
        return None

    if not os.path.exists(pdf_path):
        print("No se encontró db.pdf, no se creará índice remoto.")
        return None

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    index = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        try:
            emb = compute_embedding_remote(text)
        except Exception as e:
            print(f"Fallo al calcular embedding para chunk {i}: {e}")
            emb = []
        index.append({"id": i, "page_content": text, "embedding": emb})

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    print(f"Índice remoto creado en {index_path} con {len(index)} chunks")
    return index


def load_remote_index(index_path="vectorstore/remote_index.json"):
    if not os.path.exists(index_path):
        return None
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"No se pudo cargar índice remoto: {e}")
        return None


def cosine_similarity(a, b):
    if not a or not b:
        return -1
    if len(a) != len(b):
        return -1
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return -1
    return dot / (norm_a * norm_b)


def similarity_search_remote(query: str, k: int = 3, index_path="vectorstore/remote_index.json"):
    index = load_remote_index(index_path)
    if not index:
        # try to build index if pdf exists
        index = build_remote_index_from_pdf()
        if not index:
            return []

    try:
        q_emb = compute_embedding_remote(query)
    except Exception as e:
        print(f"Error computing query embedding: {e}")
        return []

    scored = []
    for item in index:
        emb = item.get("embedding") or []
        score = cosine_similarity(q_emb, emb)
        scored.append((score, item))

    scored = [s for s in scored if s[0] is not None and s[0] > -0.9]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [it for _, it in scored[:k]]
    return results


def load_or_create_vector_db(pdf_path="db.pdf", db_dir="vector_db"):
    # If remote embeddings selected, we'll use JSON-backed index
    if USE_REMOTE_EMBEDDINGS:
        idx = load_remote_index()
        if idx:
            print("Cargando índice remoto existente...")
            return {"type": "remote", "index_path": "vectorstore/remote_index.json"}
        # try to build
        built = build_remote_index_from_pdf(pdf_path, index_path="vectorstore/remote_index.json")
        if built:
            return {"type": "remote", "index_path": "vectorstore/remote_index.json"}
        return None

    # Otherwise try to use FAISS (lazy imports handled inside)
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
    except Exception as e:
        print(f"No se pudieron importar dependencias para FAISS: {e}")
        return None

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(db_dir):
        print("Cargando base semántica FAISS existente...")
        return FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(pdf_path):
        print("No se encontró db.pdf, continuando sin base.")
        return None

    print("Creando nueva base semántica FAISS desde db.pdf...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(db_dir)
    print("Base FAISS creada y guardada.")
    return db


def ensure_pdf_db_loaded():
    """Carga la base semántica bajo demanda (lazy)."""
    global pdf_db
    if pdf_db is None:
        pdf_db = load_or_create_vector_db()
    return pdf_db

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
