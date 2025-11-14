
from datetime import datetime, timezone


def timestamp_to_date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat()