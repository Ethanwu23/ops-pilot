import asyncio
import os

from client import AgentClient
from core import settings
from schema import ChatMessage


def get_access_token() -> str:
    """读取当前用户访问令牌。"""
    access_token = os.getenv("AGENT_ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError("AGENT_ACCESS_TOKEN is not configured")
    return access_token


async def amain() -> None:
    #### ASYNC ####
    client = AgentClient(
        settings.BASE_URL,
        access_token=get_access_token(),
    )

    print("Agent info:")
    print(client.info)

    print("Chat example:")
    response = await client.ainvoke("Tell me a brief joke?", model="gpt-5-nano")
    response.pretty_print()

    print("\nStream example:")
    async for message in client.astream("Share a quick fun fact?"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"ERROR: Unknown type - {type(message)}")


def main() -> None:
    #### SYNC ####
    client = AgentClient(
        settings.BASE_URL,
        access_token=get_access_token(),
    )

    print("Agent info:")
    print(client.info)

    print("Chat example:")
    response = client.invoke("Tell me a brief joke?", model="gpt-5-nano")
    response.pretty_print()

    print("\nStream example:")
    for message in client.stream("Share a quick fun fact?"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"ERROR: Unknown type - {type(message)}")


if __name__ == "__main__":
    print("Running in sync mode")
    main()
    print("\n\n\n\n\n")
    print("Running in async mode")
    asyncio.run(amain())
