from dataclasses import dataclass

from model.model_2 import Person


@dataclass(frozen=True, slots=True)
class Message:
    command: str
    data: dict[int, Person] | None = None


class Command:
    START = "start"
    STOP = "stop"
    DETECT = "detect"
    SEND_NEW_PERSONS = "send_new_persons"
    ANS_NEW_PERSONS = "ans_new_persons"
