from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
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
from faster_whisper import WhisperModel

load_dotenv()


WEBHOOK_SECRET=os.getenv("SECRET_KEY","xxxxx")

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
    print("Headers:", dict(request.headers))
    form = await request.form()
    file = form.get("file")
    model = form.get("model")
    prompt = form.get("prompt")
    language = form.get("language")

    if not file:
        raise HTTPException(status_code=400, detail="No 'file' part in form-data")

    # Read uploaded audio bytes
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    # Persist to a temporary file so ffmpeg can decode it
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

    # Lazy-load a single Whisper model instance (small, CPU, int8 by default)
    @lru_cache(maxsize=1)
    def get_whisper_model():
        model_size = os.getenv("WHISPER_MODEL_SIZE", "tiny")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        return WhisperModel(model_size, device=device, compute_type=compute_type)

    try:
        model_instance = get_whisper_model()
        segments, info = model_instance.transcribe(
            tmp_path,
            language=language or None,
            initial_prompt=prompt or None,
            beam_size=1,
        )
        # Collect text from segments
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

    # Match OpenAI response schema minimal field used upstream
    return JSONResponse(content={"text": transcript_text}, status_code=200)


@app.post("/attendee/webhook")
async def attendee_webhook(request: Request):
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

    # Signature is valid; return success response
    return JSONResponse(
        content={"message": "Webhook received successfully"},
        status_code=200
    )


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
            # Actually parse text message as JSON
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
                wf.setsampwidth(2)       # assume 16‑bit samples
                wf.setframerate(sample_rate)
                wf.writeframes(buffer)
            print(f"Saved audio to {fname}")
        print("WebSocket disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)