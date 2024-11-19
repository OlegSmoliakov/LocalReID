import asyncio
import logging

import zmq
from zmq.asyncio import Context

from src.base import Command, Message
from src.model import ObjectTracking

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

    detector = ObjectTracking(source, "src/weights/yolov8n.pt", out_path)

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
        msgs = []

        # receive messages from server
        response: list[Message] = await socket.recv_pyobj()

        for message in response:
            command = message.command
            log.debug(f"Received `{Command.get_name(command)}` command")

            match command:
                case Command.ANS_NEW_PERSONS:
                    if changes := detector.add_new_persons(message.data):
                        msgs.append(Message(Command.ANS_NEW_PERSONS, changes))
                case Command.SEND_NEW_PERSONS:
                    log.debug(f"Check {[message.data.keys()]} ids among detected")
                    changes = detector.check_among_detected(message.data)
                    msgs.append(Message(Command.ANS_NEW_PERSONS, changes))
                case Command.STOP:
                    exit()

        # process frame
        new_persons = detector.process_frame()
        if new_persons is None:
            await socket.send_pyobj({Message(Command.STOP)})
            log.debug("Sent `STOP` command")
        elif new_persons:
            msgs.append(Message(Command.SEND_NEW_PERSONS, new_persons))

        # send messages to server
        await socket.send_pyobj(msgs)
        for msg in msgs:
            log.debug(f"Sent `{Command.get_name(msg.command)}` command")

        # log.debug(f"Current frame: {detector.tracker.frame_no}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(levelname)s: %(message)s")
    asyncio.run(client())
