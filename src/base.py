from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Message:
    command: int
    data: Any = None


class Command:
    START = 0
    STOP = 1
    SEND_NEW_PERSONS = 2
    ANS_SIM_MAP = 3
    ANS_ADD_NEW_PERSONS = 4

    _value_to_name = {
        0: "START",
        1: "STOP",
        2: "SEND_NEW_PERSONS",
        3: "ANS_SIM_MAP",
        4: "ANS_ADD_NEW_PERSONS",
    }

    @classmethod
    def get_name(cls, value):
        return cls._value_to_name.get(value, None)
