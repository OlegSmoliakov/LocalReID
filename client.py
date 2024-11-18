import asyncio
import logging

import zmq
from zmq.asyncio import Context

from model.base import Command, Message
from model.model import ObjectTracking

# SERVER_URL = "tcp://192.168.1.2:5555"
SERVER_URL = "tcp://MacBook:5555"


log = logging.getLogger(__name__)
ctx = Context().instance()
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def init_detector():
    # source = 1
    source = "draft/campus4-c2.avi"

    # out_path = "output"
    out_path = None

    detector = ObjectTracking(source, "model/yolov8n.pt", out_path)

    return detector


async def client():
    log.info("INitializing detector")
    detector = init_detector()

    socket = ctx.socket(zmq.PAIR)
    log.info(f"Attempt to connect to server at {SERVER_URL}")
    socket.connect(SERVER_URL)
    log.info(f"Successfully connected to server at {SERVER_URL}")

    log.info("Waiting message from server")

    while True:
        response: list[Message] = await socket.recv_pyobj()

        msg_2 = None
        for message in response:
            log.debug(f"Received `{message.command}` command")
            match message.command:
                case Command.DETECT:
                    pass
                case Command.ANS_NEW_PERSONS:
                    if changes := detector.add_new_persons(message.data):
                        msg_2 = Message(Command.ANS_NEW_PERSONS, changes)
                        log.debug("Send `ans_new_persons` command")
                case Command.SEND_NEW_PERSONS:
                    changes = detector.check_among_detected(message.data)
                    msg_2 = Message(Command.ANS_NEW_PERSONS, changes)
                    log.debug("Send `ans_new_persons` command")
                case Command.STOP:
                    exit()

        new_persons = detector.process_frame()
        if new_persons is None:
            await socket.send_pyobj({Message(Command.STOP)})
        elif new_persons:
            msg_1 = Message(Command.SEND_NEW_PERSONS, new_persons)
            log.debug("Send `send_new_persons` command")
        else:
            msg_1 = None

        msg = [message for message in (msg_1, msg_2) if message]
        await socket.send_pyobj(msg)

        log.debug(f"Current frame: {detector.tracker.frame_no}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(levelname)s: %(message)s")
    asyncio.run(client())
