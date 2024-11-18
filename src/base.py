from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Message:
    command: str
    data: Any = None


class Command:
    START = 0
    STOP = 1
    DETECT = 2
    SEND_NEW_PERSONS = 3
    ANS_NEW_PERSONS = 4
