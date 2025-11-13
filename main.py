
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.post("/attendee/webhook")
async def attendee_webhook(req: Request):
    """Log webhook payloads so they are easy to inspect."""
    print(await req.json())
    return JSONResponse(content={"message": "Hello, World!"})


@app.websocket("/ws")
async def attendee_ws(websocket: WebSocket):
    """Show any incoming websocket message and echo it back."""
    await websocket.accept()
    try:
        while True:
            incoming_text = await websocket.receive_text()
            print(f"[websocket] {incoming_text}")
            await websocket.send_json({"echo": incoming_text})
    except WebSocketDisconnect:
        print("[websocket] disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)