import json
import os
import tempfile
from logging import getLogger

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.models import AttendeeEvent, get_db, init_db
from tools.ai import get_whisper_model, summarize_text
from utils.date_utils import timestamp_to_date
from utils.request_utils import serialize_optional_json, sign_payload

logger = getLogger(__name__)

load_dotenv()




WEBHOOK_SECRET=os.getenv("SECRET_KEY","xxxxx")

logger.info("WEBHOOK_SECRET =", WEBHOOK_SECRET[:3]+"..."+WEBHOOK_SECRET[-3:])



app = FastAPI()




@app.post("/debug/audio/transcriptions")
async def debug_endpoint(request: Request):
    logger.info("Getting request")
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


    try:
        model_instance = get_whisper_model()
        logger.info("transcribing...")
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
        
    logger.info("transcript_text =", transcript_text)

    return JSONResponse(content={"text": transcript_text}, status_code=200)


@app.post("/attendee/webhook")
async def attendee_webhook(request: Request, db: Session = Depends(get_db)):
    raw_body = await request.body()

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    signature_from_header = request.headers.get("X-Webhook-Signature")
    if not signature_from_header:
        raise HTTPException(status_code=400, detail="Signature missing in header")

    signature_from_payload = sign_payload(payload, WEBHOOK_SECRET)

    if signature_from_header != signature_from_payload:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for key in ("idempotency_key", "trigger"):
        if not payload.get(key):
            raise HTTPException(status_code=400, detail=f"Missing `{key}` in payload")
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
        raise HTTPException(status_code=400, detail=f"Failed to store payload: {exc}")

    return JSONResponse(
        content={"message": "Webhook received successfully"},
        status_code=200
    )


@app.get("/attendee/{bot_id}/summary")
async def attendee_summary(bot_id: str, db: Session = Depends(get_db)):
    records = (
        db.query(AttendeeEvent)
        .filter(AttendeeEvent.bot_id == bot_id)
        .order_by(AttendeeEvent.timestamp_ms.asc())
        .all()
    )
    if not records:
        raise HTTPException(status_code=404, detail="No events found for this bot.")

    transcripts = [
        {
            "speaker_name": row.speaker_name,
            "speaker_uuid": row.speaker_uuid,
            "timestamp": timestamp_to_date(row.timestamp_ms),
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

        conversation = "\n".join(
            f"{item['speaker_name'] or 'Unknown'}: {item['text']}"
            for item in transcripts
        )
        summary = summarize_text(conversation)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to summarize transcript: {exc}")

    return {"bot_id": bot_id, "summary": summary, "transcripts": transcripts}



@app.on_event("startup")
def _startup_db():
    init_db()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, reload=True)