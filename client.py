import asyncio

import zmq
from zmq.asyncio import Context

SERVER_URL = "tcp://192.168.1.3:5555"

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
ctx = Context().instance()


async def client():
    socket = ctx.socket(zmq.PAIR)
    socket.connect(SERVER_URL)
    for i in range(10):
        data = {"text": f"this is a test №{i}"}
        socket.send_json(data)
        print(f"sent {data}")

        response = await socket.recv_json()
        print(f"received {response}")

        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(client())
