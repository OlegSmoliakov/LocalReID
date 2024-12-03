import asyncio
import logging
import os
import time

import zmq
from zmq.asyncio import Context

from src.base import Command, Message
from src.model import ObjectTracking

URL = "tcp://*:5555"

log = logging.getLogger(__name__)
ctx = Context.instance()


def init_detector():
    # source = 1
    source = "draft/campus4-c0.avi"

    out_path = "output_" + os.path.basename(source).split(".")[0]
    # out_path = None

    detector = ObjectTracking(source, output_video=out_path)

    return detector


async def server():
    log.info("Initializing detector...")
    detector = init_detector()

    socket = ctx.socket(zmq.PAIR)
    socket.bind(URL)
    log.info(f"Server started at {URL}")

    log.info("Waiting for client...")

    # skip first N frames
    N = 47
    # N = 40
    for _ in range(N):
        # detector.process_frame()
        detector.cap.read()

    msgs: list[Message] = []
    await socket.send_pyobj(msgs)

    while True:
        # process frame
        start_time = time.time()
        new_persons = detector.process_frame()
        if new_persons is None:
            await socket.send_pyobj({Message(Command.STOP)})
            log.debug("Video stopped, sent `STOP` command")
            break
        elif new_persons:
            msgs.append(Message(Command.SEND_NEW_PERSONS, new_persons))
        # log.debug(f"Process frame took: {time.time() - start_time:.4f}")

        # send messages
        start_time = time.time()
        await socket.send_pyobj(msgs)
        # log.debug(f"Sent took: {time.time() - start_time:.4f}")
        for msg in msgs:
            log.debug(f"Sent `{Command.get_name(msg.command)}` command")
        msgs = []

        # receive messages
        start_time = time.time()  # noqa: F841
        response: list[Message] = await socket.recv_pyobj()
        # log.debug(f"Received took: {time.time() - start_time:.4f}")
        for message in response:
            log.debug(f"Received `{Command.get_name(message.command)}` command")
            match message.command:
                case Command.ANS_ADD_NEW_PERSONS:
                    if changes := detector.add_new_persons(message.data):
                        msgs.append(Message(Command.ANS_ADD_NEW_PERSONS, changes))
                case Command.ANS_SIM_MAP:
                    if changes := detector.add_new_persons(message.data):
                        msgs.append(Message(Command.ANS_ADD_NEW_PERSONS, changes))
                case Command.SEND_NEW_PERSONS:
                    log.debug(f"Check {[idx for idx in message.data]} ids among detected")
                    changes = detector.check_among_detected(message.data)
                    msgs.append(Message(Command.ANS_SIM_MAP, changes))
                case Command.STOP:
                    detector.release_resources()
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
