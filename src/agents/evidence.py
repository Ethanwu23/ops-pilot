import json

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from schema.assets import AssetLookupInput, AssetResponse
from schema.diagnosis import Evidence
from schema.metrics import MetricsLookupInput, MetricsResponse


def collect_tool_messages(
    messages: list[BaseMessage],
) -> tuple[AIMessage, list[ToolMessage]]:
    """提取消息末尾本轮工具结果，并核对请求与结果的对应关系。"""
    index = len(messages)

    while index > 0 and isinstance(messages[index - 1], ToolMessage):
        index -= 1

    tool_messages = messages[index:]

    if not tool_messages:
        raise ValueError("消息末尾没有本轮工具结果")

    if index == 0:
        raise ValueError("工具结果前缺少模型请求消息")

    request_message = messages[index - 1]

    if not isinstance(request_message, AIMessage):
        raise ValueError("工具结果前的消息必须是 AIMessage")

    if not request_message.tool_calls:
        raise ValueError("模型消息没有工具请求")

    request_ids = [call["id"] for call in request_message.tool_calls]
    result_ids = [message.tool_call_id for message in tool_messages]

    if any(not call_id for call_id in request_ids + result_ids):
        raise ValueError("工具调用 ID 不能为空")

    if len(request_ids) != len(set(request_ids)):
        raise ValueError("工具请求 ID 重复")

    if len(result_ids) != len(set(result_ids)):
        raise ValueError("工具结果 ID 重复")

    if set(request_ids) != set(result_ids):
        raise ValueError("工具请求与结果的 ID 不匹配")

    return request_message, tool_messages


def extract_evidence(messages: list[BaseMessage]) -> list[Evidence]:
    """校验本轮资产和指标工具结果，生成保留原始字段值的证据。"""
    request_message, tool_messages = collect_tool_messages(messages)

    # 按调用 ID 配对，不依赖工具结果的返回顺序。
    calls_by_id = {call["id"]: call for call in request_message.tool_calls}

    evidence_list: list[Evidence] = []

    for message in tool_messages:
        call = calls_by_id[message.tool_call_id]

        if message.status == "error":
            raise ValueError(f"工具执行失败，不能生成证据：{message.tool_call_id}")

        if not isinstance(message.content, str):
            raise ValueError("工具结果内容必须是 JSON 字符串")

        data = json.loads(message.content)

        if not isinstance(data, dict):
            raise ValueError("工具结果 JSON 必须是对象")

        if call["name"] == "get_metrics":
            # 严格检查原始布尔值，避免数字 1 被解析成 True。
            if data.get("is_test_data") is not True:
                raise ValueError("指标证据必须明确标记为测试数据")

            metrics_request = MetricsLookupInput.model_validate(call["args"])
            metrics = MetricsResponse.model_validate(data)

            if metrics.asset_id != metrics_request.asset_id:
                raise ValueError(
                    f"指标证据资产 ID 不匹配："
                    f"请求 {metrics_request.asset_id}，"
                    f"返回 {metrics.asset_id}"
                )

            source = metrics.source
            is_test_data = metrics.is_test_data

        elif call["name"] == "get_asset":
            asset_request = AssetLookupInput.model_validate(call["args"])

            if set(data) != {"source", "found", "asset"}:
                raise ValueError("资产工具结果字段不符合约定")

            if data["source"] != "local_learning_database":
                raise ValueError("资产证据来源不符合约定")

            if type(data["found"]) is not bool:
                raise ValueError("资产查询 found 必须是布尔值")

            if data["found"]:
                asset = AssetResponse.model_validate(data["asset"])

                if asset.id != asset_request.asset_id:
                    raise ValueError(
                        f"资产证据 ID 不匹配：请求 {asset_request.asset_id}，返回 {asset.id}"
                    )
            elif data["asset"] is not None:
                raise ValueError("未找到资产时，asset 必须为 None")

            # 实际登记查询不等于模拟指标，也不代表实时健康观测。
            source = data["source"]
            is_test_data = False

        else:
            raise ValueError(f"尚不支持提取该工具的证据：{call['name']}")

        # 保留解析后的原始数据，不重新生成日期、IP 或指标字段。
        evidence = Evidence(
            evidence_id=message.tool_call_id,
            tool_name=call["name"],
            source=source,
            is_test_data=is_test_data,
            data=data,
        )
        evidence_list.append(evidence)

    return evidence_list
