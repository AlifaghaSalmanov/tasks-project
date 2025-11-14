from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import base64
import wave
import asyncio
from dotenv import load_dotenv
import os
import hmac
import hashlib
import json
import tempfile
from functools import lru_cache
from datetime import datetime, timezone
from faster_whisper import WhisperModel
from sqlalchemy.orm import Session
import requests
from db import init_db, get_db, AttendeeEvent, serialize_optional_json
from ai import summarize_text

load_dotenv()

# Optional remote debugging with debugpy (enabled only if env DEBUGPY=1)
DEBUGPY_ENABLED = os.getenv("DEBUGPY", "0") == "1"
if DEBUGPY_ENABLED:
    try:
        import debugpy
        host = os.getenv("DEBUGPY_HOST", "0.0.0.0")
        port = int(os.getenv("DEBUGPY_PORT", "5679"))
        debugpy.listen((host, port))
        print(f"[debugpy] Listening on {host}:{port}")
        if os.getenv("DEBUGPY_WAIT_FOR_CLIENT", "0") == "1":
            print("[debugpy] Waiting for debugger to attach...")
            debugpy.wait_for_client()
    except Exception as e:
        print(f"[debugpy] Failed to initialize: {e}")


WEBHOOK_SECRET=os.getenv("SECRET_KEY","xxxxx")
BOT_API_BASE = os.getenv("BOT_API_BASE", "http://localhost:8000")
BOT_API_TOKEN = os.getenv("BOT_API_TOKEN", "")

print(WEBHOOK_SECRET[:3]+"..."+WEBHOOK_SECRET[-3:])


def sign_payload(payload: dict, secret_b64: str) -> str:
    """
    Sign a webhook payload using HMAC-SHA256.
    Returns a base64-encoded HMAC-SHA256 signature.
    """
    # Convert payload into a canonical JSON string (stable keys, no spaces)
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    # Secret provided by the dashboard is base64-encoded; decode to raw bytes
    secret_decoded = base64.b64decode(secret_b64)
    # Compute binary HMAC-SHA256 digest over the canonical JSON string
    signature_bytes = hmac.new(
        secret_decoded,
        payload_json.encode("utf-8"),
        hashlib.sha256
    ).digest()
    # Return base64-encoded signature string
    return base64.b64encode(signature_bytes).decode("utf-8")

app = FastAPI()

@app.on_event("startup")
def _startup_db():
    # Ensure SQLite tables exist before handling requests
    init_db()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Print concise reason for 400 errors to server logs
    if exc.status_code == 400:
        print(f"[HTTP 400] {request.method} {request.url.path} - {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.post("/debug/audio/transcriptions")
async def debug_endpoint(request: Request):
    """
    Debug endpoint that mimics OpenAI's /audio/transcriptions.
    It accepts multipart/form-data with fields: file, model, optional prompt, language.
    """
    print("Getting request")
    form = await request.form()
    file = form.get("file")
    model = form.get("model")
    prompt = form.get("prompt")
    language = form.get("language")

    if not file:
        raise HTTPException(status_code=400, detail="No 'file' part in form-data")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    suffix = ""
    filename = getattr(file, "filename", "") or ""
    if "." in filename:
        suffix = "." + filename.split(".")[-1].lower()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store temp audio file: {e}")

    @lru_cache(maxsize=1)
    def get_whisper_model():
        model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        return WhisperModel(model_size, device=device, compute_type=compute_type)

    try:
        model_instance = get_whisper_model()
        print("transcribing...")
        segments, info = model_instance.transcribe(
            tmp_path,
            language=language or None,
            initial_prompt=prompt or None,
            beam_size=1,
        )

        text_parts = []
        for seg in segments:
            text_parts.append(seg.text)
        transcript_text = "".join(text_parts).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        
    print("transcript_text =", transcript_text)

    # Match OpenAI response schema minimal field used upstream
    return JSONResponse(content={"text": transcript_text}, status_code=200)


@app.post("/attendee/webhook")
async def attendee_webhook(request: Request, db: Session = Depends(get_db)):
    raw_body = await request.body()
    # Parse the JSON body
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    print("Received payload =", payload)

    # Get signature from header
    signature_from_header = request.headers.get("X-Webhook-Signature")
    if not signature_from_header:
        raise HTTPException(status_code=400, detail="Signature missing in header")

    # Compute expected signature using the shared secret
    signature_from_payload = sign_payload(payload, WEBHOOK_SECRET)
    print("signature_from_header =", signature_from_header)
    print("signature_from_payload =", signature_from_payload)

    if signature_from_header != signature_from_payload:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Persist the event once the signature is verified.
    for key in ("idempotency_key", "trigger"):
        if not payload.get(key):
            raise HTTPException(status_code=400, detail=f"Missing `{key}` in payload")
    # Unpack nested attendee data for storage.
    data_block = payload.get("data") or {}
    transcription = (data_block.get("transcription") or {}).get("transcript")

    event = AttendeeEvent(
        idempotency_key=payload["idempotency_key"],
        trigger=payload["trigger"],
        bot_id=payload.get("bot_id"),
        bot_metadata=serialize_optional_json(payload.get("bot_metadata")),
        duration_ms=data_block.get("duration_ms"),
        speaker_name=data_block.get("speaker_name"),
        speaker_uuid=data_block.get("speaker_uuid"),
        timestamp_ms=data_block.get("timestamp_ms"),
        transcript_text=transcription,
        speaker_is_host=data_block.get("speaker_is_host"),
        speaker_user_uuid=data_block.get("speaker_user_uuid"),
    )
    try:
        db.add(event)
        db.commit()
    except Exception as exc:
        db.rollback()
        # Propagate DB errors to the client for easy debugging.
        raise HTTPException(status_code=400, detail=f"Failed to store payload: {exc}")

    # Signature is valid; return success response
    return JSONResponse(
        content={"message": "Webhook received successfully"},
        status_code=200
    )


@app.get("/attendee/{bot_id}/summary")
async def attendee_summary(bot_id: str, db: Session = Depends(get_db)):
    """Summarize all transcript text stored for a given bot."""
    records = (
        db.query(AttendeeEvent)
        .filter(AttendeeEvent.bot_id == bot_id)
        .order_by(AttendeeEvent.timestamp_ms.asc())
        .all()
    )
    if not records:
        raise HTTPException(status_code=404, detail="No events found for this bot.")

    # Collect every utterance so the client can see who said what and when.
    transcripts = [
        {
            "speaker_name": row.speaker_name,
            "speaker_uuid": row.speaker_uuid,
            "timestamp": datetime.fromtimestamp(row.timestamp_ms / 1000, tz=timezone.utc).isoformat()
            if row.timestamp_ms is not None
            else None,
            "duration_ms": row.duration_ms,
            "text": row.transcript_text,
            "is_host": row.speaker_is_host,
        }
        for row in records
        if row.transcript_text
    ]
    if not transcripts:
        raise HTTPException(status_code=400, detail="No transcript text available to summarize.")

    try:
        # Feed speaker labels to AI so summary keeps names.
        conversation = "\n".join(
            f"{item['speaker_name'] or 'Unknown'}: {item['text']}"
            for item in transcripts
        )
        summary = summarize_text(conversation)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to summarize transcript: {exc}")

    return {"bot_id": bot_id, "summary": summary, "transcripts": transcripts}


def _bot_headers() -> dict:
    if not BOT_API_TOKEN:
        raise HTTPException(status_code=500, detail="Missing `BOT_API_TOKEN` configuration.")
    return {
        "Authorization": f"Token {BOT_API_TOKEN}",
        "Content-Type": "application/json",
    }


@app.post("/bots")
async def create_bot_endpoint(body: dict):
    """Proxy request to upstream bot creation service."""
    upstream_url = f"{BOT_API_BASE}/api/v1/bots"
    try:
        response = requests.post(upstream_url, headers=_bot_headers(), json=body, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Bot service error: {exc}")
    return response.json()


@app.post("/bots/{bot_id}/leave")
async def leave_bot_endpoint(bot_id: str):
    """Instruct upstream service to remove the bot from its meeting."""
    upstream_url = f"{BOT_API_BASE}/api/v1/bots/{bot_id}/leave"
    try:
        response = requests.post(upstream_url, headers=_bot_headers(), timeout=30)
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Bot service error: {exc}")
    return response.json()


@app.websocket("/ws")
async def attendee_ws(ws: WebSocket):
    await ws.accept()
    buffer = bytearray()
    sample_rate = None
    try:
        while True:
            msg = await ws.receive_text()
            obj = await ws.receive_json() if False else None  # placeholder
            print(msg)
            data = __import__("json").loads(msg)
            if data.get("trigger") == "realtime_audio.mixed":
                chunk_b64 = data["data"]["chunk"]
                sample_rate = data["data"]["sample_rate"]
                decoded = base64.b64decode(chunk_b64)
                buffer.extend(decoded)
    except WebSocketDisconnect:
        if buffer and sample_rate:
            fname = "meeting_audio.wav"
            with wave.open(fname, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(buffer)
            print(f"Saved audio to {fname}")
        print("WebSocket disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, reload=True)