import asyncio
import logging
import time
import zlib

import numpy as np
import zmq
from pyinstrument import Profiler
from zmq.asyncio import Context

from src.base import Command, Message
from src.model import ObjectTracking, Person

URL = "tcp://*:5555"

log = logging.getLogger(__name__)
ctx = Context.instance()


def init_detector():
    # source = 1
    source = "draft/campus4-c0.avi"

    out_path = "output"
    # out_path = None

    detector = ObjectTracking(source, "src/weights/yolov8n.pt", out_path)

    return detector


async def server():
    log.info("Initializing detector...")
    detector = init_detector()

    socket = ctx.socket(zmq.PAIR)
    socket.bind(URL)
    log.info(f"Server started at {URL}")

    log.info("Waiting for client...")

    msg_1, msg_2 = None, None

    # skip first N frames
    N = 47
    for _ in range(N):
        detector.process_frame()

    while True:
        msg = [message for message in (msg_1, msg_2) if message]

        start_time = time.time()
        await socket.send_pyobj(msg)
        # log.debug(f"Sent took: {time.time() - start_time:.4f}")

        start_time = time.time()
        new_persons = detector.process_frame()
        if new_persons is None:
            await socket.send_pyobj({Message(Command.STOP)})
            break
        elif new_persons:
            msg_1 = Message(Command.SEND_NEW_PERSONS, new_persons)
        else:
            msg_1 = None
        # log.debug(f"Process frame took: {time.time() - start_time:.4f}")

        start_time = time.time()
        response: list[Message] = await socket.recv_pyobj()
        # log.debug(f"Received took: {time.time() - start_time:.4f}")

        msg_2 = None
        for message in response:
            log.debug(f"Received `{message.command}` command")
            match message.command:
                case Command.DETECT:
                    pass
                case Command.ANS_NEW_PERSONS:
                    if changes := detector.add_new_persons(message.data):
                        msg_2 = Message(Command.ANS_NEW_PERSONS, changes)
                case Command.SEND_NEW_PERSONS:
                    changes = detector.check_among_detected(message.data)
                    msg_2 = Message(Command.ANS_NEW_PERSONS, changes)
                case Command.STOP:
                    exit()

        # log.debug(f"Current frame: {detector.tracker.frame_no}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(levelname)s: %(message)s")
    asyncio.run(server())


# serialized_persons = {
#     idx: Person(person.track, zlib.compress(person.img.tobytes()))
#     for idx, person in persons.items()
# }

# persons = {
#     idx: Person(person.track, np.frombuffer(zlib.decompress(person.img)))
#     for idx, person in response_msg.data.items()
# }
