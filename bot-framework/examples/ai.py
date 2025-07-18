# WARNING: This might not work in the future. Do NOT use this in production.

import asyncio
import socketio
from env import WEBUI_URL, TOKEN
from utils import send_message, send_typing


MODEL_ID = "llama3.2:latest"

# Create an asynchronous Socket.IO client instance
sio = socketio.AsyncClient(logger=False, engineio_logger=False)


# Event handlers
@sio.event
async def connect():
    print("Connected!")


@sio.event
async def disconnect():
    print("Disconnected from the server!")


import aiohttp
import asyncio


async def openai_chat_completion(messages):
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{WEBUI_URL}/api/chat/completions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json=payload,
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                # Optional: Handle errors or return raw response text
                return {"error": await response.text(), "status": response.status}


# Define an async function for the main workflow
async def main():
    try:
        print(f"Connecting to {WEBUI_URL}...")
        await sio.connect(
            WEBUI_URL, socketio_path="/ws/socket.io", transports=["websocket"]
        )
        print("Connection established!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Add debugging for all socket events
    @sio.event
    async def connect():
        print("Socket connected!")

    @sio.event
    async def disconnect():
        print("Socket disconnected!")

    # Set up event handlers for all users (simplified approach)
    @sio.on("channel-events")
    async def channel_events(data):
        print(f"Received channel event: {data}")
        
        if data["data"]["type"] == "message":
            print(f'{data["user"]["name"]}: {data["data"]["data"]["content"]}')
            await send_typing(sio, data["channel_id"])

            async def send_typing_until_complete(channel_id, coro):
                """
                Sends typing indicators every second until the provided coroutine completes.
                """
                task = asyncio.create_task(coro)  # Begin the provided coroutine task
                try:
                    # While the task is running, send typing indicators every second
                    while not task.done():
                        await send_typing(sio, channel_id)
                        await asyncio.sleep(1)
                    # Await the actual result of the coroutine
                    return await task
                except Exception as e:
                    task.cancel()
                    raise e  # Propagate any exceptions that occurred in the coroutine

            # OpenAI API coroutine
            openai_task = openai_chat_completion(
                [
                    {"role": "system", "content": "You are a friendly AI assistant."},
                    {"role": "user", "content": data["data"]["data"]["content"]},
                ]
            )

            try:
                # Run OpenAI coroutine while showing typing indicators
                response = await send_typing_until_complete(
                    data["channel_id"], openai_task
                )

                if response.get("choices"):
                    completion = response["choices"][0]["message"]["content"]
                    await send_message(data["channel_id"], completion)
                else:
                    await send_message(
                        data["channel_id"], "I'm sorry, I don't understand."
                    )
            except Exception as e:
                print(f"Error processing message: {e}")
                await send_message(
                    data["channel_id"],
                    "Something went wrong while processing your request.",
                )

    # Add a catch-all event handler to see what events we're getting
    @sio.event
    async def catch_all(event, *args):
        print(f"Received event '{event}' with args: {args}")

    # Authenticate with the server and join channels
    try:
        print("Authenticating...")
        result = await sio.emit("user-join", {"auth": {"token": TOKEN}})
        print(f"Authentication result: {result}")
        
        # Also try to join channels explicitly
        print("Joining channels...")
        await sio.emit("join-channels", {"auth": {"token": TOKEN}})
        print("Channel join request sent")
        
    except Exception as e:
        print(f"Authentication error: {e}")

    print("Bot is ready and listening for events...")
    
    # Wait indefinitely to keep the connection open
    await sio.wait()


# Actually run the async `main` function using `asyncio`
if __name__ == "__main__":
    asyncio.run(main())
