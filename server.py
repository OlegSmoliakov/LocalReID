import asyncio

import zmq
from zmq.asyncio import Context

URL = "tcp://*:5555"
ctx = Context.instance()


async def server():
    socket = ctx.socket(zmq.PAIR)
    socket.bind(URL)
    while True:
        message = await socket.recv_json()
        print(f"Received: {message}")

        response = {"data": message}
        await socket.send_json(response)
        print(f"Sent: {response}")


if __name__ == "__main__":
    asyncio.run(server())
