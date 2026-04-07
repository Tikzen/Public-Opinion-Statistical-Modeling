import json
from pathlib import Path

import requests


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "agent_config.json"


def load_config(config_path=None):
    """
    读取配置文件
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_api_key_for_provider(config: dict, provider_name: str, api_key: str | None = None) -> str:
    """
    API Key 优先级：
    1. 当前 provider 下的 api_key
    2. 前端输入的 api_key
    """
    provider = config.get("providers", {}).get(provider_name, {})

    config_key = (provider.get("api_key") or "").strip()
    input_key = (api_key or "").strip()

    if config_key:
        return config_key
    if input_key:
        return input_key

    raise ValueError(f"未检测到 {provider_name} 的 API Key：请先在 agent_config.json 中配置，或通过前端输入。")


def build_messages(data_summary, config: dict) -> list:
    """
    构造消息
    """
    prompt_templates = config.get("prompt_templates", {})
    system_prompt = prompt_templates.get(
        "system_prompt",
        "你是统计建模项目中的传播数据分析智能体，请输出简洁专业的中文结论。"
    )
    task_prompt = prompt_templates.get(
        "task_prompt",
        "请根据输入的数据摘要与参数建议，生成中文分析。"
    )
    constraints = prompt_templates.get("constraints", [])

    user_prompt = (
        f"{task_prompt}\n\n"
        f"输入数据：\n{json.dumps(data_summary, ensure_ascii=False, indent=2)}\n\n"
    )

    if constraints:
        user_prompt += "约束要求：\n"
        for i, item in enumerate(constraints, start=1):
            user_prompt += f"{i}. {item}\n"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_provider(config: dict, provider_name: str, data_summary, api_key=None) -> str:
    """
    调用指定 provider
    """
    providers = config.get("providers", {})
    provider = providers.get(provider_name, {})

    if not provider:
        raise ValueError(f"未找到 provider 配置：{provider_name}")

    if provider.get("enabled", True) is False:
        raise ValueError(f"provider 已被禁用：{provider_name}")

    base_url = (provider.get("base_url") or "").strip()
    model = (provider.get("model") or "").strip()
    timeout = int(provider.get("timeout", 60))
    temperature = provider.get("temperature", 0.4)
    max_tokens = provider.get("max_tokens", 1200)

    if not base_url:
        raise ValueError(f"{provider_name} 未配置 base_url")
    if not model:
        raise ValueError(f"{provider_name} 未配置 model")

    final_api_key = resolve_api_key_for_provider(config, provider_name, api_key)
    messages = build_messages(data_summary, config)

    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"].strip()
    return f"[当前调用平台: {provider_name}]\n\n{content}"


def analyze_with_llm(data_summary, api_key=None, config_path=None) -> str:
    """
    对外统一调用入口

    自动切换策略：
    1. 优先使用 active_provider（建议设为 openai_compatible，即 DeepSeek）
    2. 若失败，则自动回退到 siliconflow
    """
    config = load_config(config_path)

    active_provider = config.get("active_provider", "openai_compatible")
    fallback_provider = "siliconflow"

    provider_order = [active_provider]
    if fallback_provider not in provider_order:
        provider_order.append(fallback_provider)

    last_error = None

    for provider_name in provider_order:
        try:
            return call_provider(config, provider_name, data_summary, api_key=api_key)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"所有可用 LLM 平台均请求失败。最后一次错误：{last_error}")
