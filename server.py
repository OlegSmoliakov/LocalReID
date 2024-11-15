import asyncio
import logging
import pickle

import zmq
from zmq.asyncio import Context

from model.model_2 import ObjectTracking, Person

URL = "tcp://*:5555"

log = logging.getLogger(__name__)
ctx = Context.instance()


def init_detector():
    source = 1
    # source = "draft/campus4-c0.avi"

    out_path = "output"
    # out_path = None

    detector = ObjectTracking(source, "model/yolov8n.pt", out_path)

    return detector


async def server():
    log.info("Initializing detector...")
    detector = init_detector()
    persons: dict[int, Person] = {}

    socket = ctx.socket(zmq.PAIR)
    socket.bind(URL)

    log.info(f"Server started at {URL}")
    log.info("Waiting for client...")

    socket.send_json({"message": "Hello from server!"})
    socket.recv_json()

    while True:
        persons = detector.process_frame(persons)
        if persons is None:
            socket.send_json({"command": "stop"})
            break

        serialized_persons = [pickle.dumps(person) for person in persons]

        socket.send_json({"command": "detect", "data": serialized_persons})
        response = await socket.recv_json()

        assert response["command"] == "detect", "Invalid command received from client"
        log.debug(f"Received command: {response["command"]}")

        persons = response["data"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(levelname)s: %(message)s")
    asyncio.run(server())
