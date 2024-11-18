from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Message:
    command: str
    data: Any = None


class Command:
    START = "start"
    STOP = "stop"
    DETECT = "detect"
    SEND_NEW_PERSONS = "send_new_persons"
    ANS_NEW_PERSONS = "ans_new_persons"
