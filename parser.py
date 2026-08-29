"""
parser.py
Parses a raw WhatsApp chat export (.txt) into structured records.

WhatsApp export line formats vary slightly by platform/locale, e.g.:
  12/1/24, 9:03 AM - John: Hey, are we still on for lunch today?
  1/12/2024, 21:03 - John: Hey there
  [12/1/24, 9:03:12 AM] John: Hey there   (iOS sometimes wraps in brackets)

Multi-line messages (no timestamp prefix) are appended to the previous message.
System messages ("X joined using this group's invite link", "Messages are
end-to-end encrypted", etc.) are tagged as message_type="system" and excluded
from downstream analysis by default.
"""

import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional


# Matches the start-of-line timestamp + sender pattern across common export formats.
# Handles optional [brackets], optional seconds, optional AM/PM, and both '-' and
# the unicode en-dash some exports use.
LINE_PATTERN = re.compile(
    r"""^\[?
    (?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*
    (?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)
    \]?\s*[-\u2013]\s*
    (?P<sender>[^:]+):\s
    (?P<message>.*)$""",
    re.VERBOSE,
)

# System messages have no ": " sender/message split (no one to attribute to).
SYSTEM_LINE_PATTERN = re.compile(
    r"""^\[?
    (?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*
    (?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)
    \]?\s*[-\u2013]\s*
    (?P<message>[^:]*)$""",
    re.VERBOSE,
)

DATE_FORMATS = [
    "%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p",
    "%d/%m/%y %I:%M %p", "%d/%m/%Y %I:%M %p",
    "%m/%d/%y %H:%M", "%m/%d/%Y %H:%M",
    "%d/%m/%y %H:%M", "%d/%m/%Y %H:%M",
    "%m/%d/%y %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p",
    "%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
]


@dataclass
class ChatMessage:
    timestamp: Optional[datetime]
    sender: Optional[str]
    message: str
    message_type: str  # "text" | "media" | "system"


def _parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    raw = f"{date_str} {time_str}".replace("\u202f", " ").strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _classify_message(text: str) -> str:
    if "<Media omitted>" in text or "image omitted" in text.lower() or "video omitted" in text.lower():
        return "media"
    return "text"


def parse_whatsapp_export(filepath: str) -> List[ChatMessage]:
    """Parse a WhatsApp .txt export into a list of ChatMessage records."""
    messages: List[ChatMessage] = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        match = LINE_PATTERN.match(line)
        if match:
            dt = _parse_datetime(match.group("date"), match.group("time"))
            sender = match.group("sender").strip()
            text = match.group("message").strip()
            messages.append(
                ChatMessage(
                    timestamp=dt,
                    sender=sender,
                    message=text,
                    message_type=_classify_message(text),
                )
            )
            continue

        sys_match = SYSTEM_LINE_PATTERN.match(line)
        if sys_match:
            dt = _parse_datetime(sys_match.group("date"), sys_match.group("time"))
            messages.append(
                ChatMessage(
                    timestamp=dt,
                    sender=None,
                    message=sys_match.group("message").strip(),
                    message_type="system",
                )
            )
            continue

        # No timestamp match -> continuation of the previous multi-line message
        if messages:
            messages[-1].message += "\n" + line.strip()

    return messages


def to_dict_list(messages: List[ChatMessage]) -> List[dict]:
    out = []
    for m in messages:
        d = asdict(m)
        d["timestamp"] = m.timestamp.isoformat() if m.timestamp else None
        out.append(d)
    return out


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_chat.txt"
    parsed = parse_whatsapp_export(path)
    print(f"Parsed {len(parsed)} messages\n")
    for m in parsed[:5]:
        print(m)
    print("\n--- JSON preview ---")
    print(json.dumps(to_dict_list(parsed)[:3], indent=2))