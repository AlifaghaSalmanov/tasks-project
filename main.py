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


@app.post("/debug")
async def debug_endpoint(request: Request):
    raw_body = await request.body()
    print("Raw body:", raw_body)
    
    # Try to parse as JSON if possible
    try:
        json_body = json.loads(raw_body)
        print("JSON body:", json_body)
    except json.JSONDecodeError:
        print("Body is not valid JSON")
    
    # Also print headers for debugging
    print("Headers:", dict(request.headers))
    
    return {"message": "Request body printed to console"}


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