import base64
import hashlib
import hmac
import json


def sign_payload(payload: dict, secret_b64: str) -> str:
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    secret_decoded = base64.b64decode(secret_b64)

    signature_bytes = hmac.new(
        secret_decoded,
        payload_json.encode("utf-8"),
        hashlib.sha256
    ).digest()

    return base64.b64encode(signature_bytes).decode("utf-8")


def serialize_optional_json(value) -> str:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
