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

load_dotenv()


WEBHOOK_SECRET=os.getenv("SECRET_KEY","xxxxx")

print(WEBHOOK_SECRET[:3])


def verify_signature(secret: str, payload: str, signature: str) -> bool:
    """
    Verify the HMAC signature of the payload.
    """
    # Calculate the HMAC using SHA-256
    calculated_signature = hmac.new(
        key=secret.encode('utf-8'),  # Webhook secret
        msg=payload.encode('utf-8'),  # The raw payload
        digestmod=hashlib.sha256
    ).hexdigest()
    
    # Compare the calculated signature with the provided signature
    return hmac.compare_digest(calculated_signature, signature)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/attendee/webhook")
async def attendee_webhook(request: Request):
    payload = await request.body()
    payload_str = payload.decode('utf-8')  # Convert to string if it's binary

    # Extract the signature from the headers (assuming the header is called 'X-Signature')
    signature = request.headers.get('X-Signature')

    if not signature:
        raise HTTPException(status_code=400, detail="Signature missing in header")

    # Verify the signature
    if not verify_signature(WEBHOOK_SECRET, payload_str, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Process the valid webhook payload here (e.g., log it, trigger further actions)
    return JSONResponse(content={
        "message": "Webhook received and signature verified successfully.",
        "received_payload": payload_str  # Print the first 100 characters of the payload for reference
    })


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