from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint
from langgraph.prebuilt import ToolNode
from langgraph.runtime import get_runtime
from agents.context import AgentContext
from agents.metrics_tool import get_metrics

from core import get_model, settings
from agents.asset_tool import get_asset


@entrypoint(context_schema=AgentContext)
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str, list[BaseMessage]],
    config: RunnableConfig,
):
    messages = inputs["messages"]
    if previous:
        messages = previous["messages"] + messages

    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    tools = [get_asset, get_metrics]
    bound_model = model.bind_tools(tools)
    response = await bound_model.ainvoke(messages)
    new_messages: list[BaseMessage] = [response]

    if response.tool_calls:
        tool_node = ToolNode(tools, handle_tool_errors=False)
        tool_result = await tool_node.ainvoke(
            {"messages": messages + new_messages},
            config=config,
            runtime=get_runtime(),
        )
        new_messages.extend(tool_result["messages"])
        final_response = await model.ainvoke(
            messages + new_messages,
            config=config,
        )
        new_messages.append(final_response)


    return entrypoint.final(
        value={"messages": new_messages}, save={"messages": messages + new_messages}
    )
