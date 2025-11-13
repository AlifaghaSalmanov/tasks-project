
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import base64
import wave
import asyncio

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post("/attendee/webhook")
async def attendee_webhook(req: Request):
    """Log webhook payloads so they are easy to inspect."""
    print(await req.json())
    return JSONResponse(content={"message": "Hello, World!"})


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