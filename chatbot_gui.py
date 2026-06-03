import json
from urllib import request
from urllib.error import HTTPError, URLError

import gradio as gr

from config import ProjectConfig
from retriever import retrieve_candidates


def _format_context(candidates: list[dict]) -> str:
    lines = []
    for index, candidate in enumerate(candidates, start=1):
        lines.append(
            f"[{index}] idiom={candidate['idiom']} | translation={candidate['translation']} | meaning={candidate['meaning']}"
        )
    return "\n".join(lines)


def _format_history(history: list[dict], max_turns: int) -> str:
    if not history or max_turns <= 0:
        return ""

    recent = history[-(max_turns * 2) :]
    lines: list[str] = []
    for item in recent:
        if isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
            lines.append(f"{role}: {content}")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lines.append(f"user: {item[0]}")
            lines.append(f"assistant: {item[1]}")
    return "\n".join(lines)


def _call_chat_api(message: str, history: list[dict], candidates: list[dict], config: ProjectConfig) -> str:
    config.validate_chat_config()

    context = _format_context(candidates)
    history_text = _format_history(history, config.chat_history_turns)
    system_prompt = (
        "You are an idiom translation assistant. Use the retrieved idiom context first. "
        "If exact idiom is missing, provide best explanation and closest equivalent."
    )
    user_prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"User message: {message}\n\n"
        f"Retrieved idioms:\n{context}\n\n"
        "Return concise, practical answer."
    )

    payload = {
        "model": config.chat_api_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.chat_api_key:
        headers["Authorization"] = f"Bearer {config.chat_api_key}"

    req = request.Request(
        url=config.chat_api_url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.chat_api_timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Chat API error ({error.code}): {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Chat API connection failed: {error}") from error

    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("answer"), str):
            return parsed["answer"]
        if isinstance(parsed.get("output_text"), str):
            return parsed["output_text"]
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message_obj = first.get("message")
                if isinstance(message_obj, dict) and isinstance(message_obj.get("content"), str):
                    return message_obj["content"]

    raise RuntimeError("Unsupported chat API response format.")


def _fallback_response(message: str, candidates: list[dict]) -> str:
    if not candidates:
        return "I could not find a close idiom match in the current index."

    best = candidates[0]
    lines = [
        f"Best match: {best['idiom']}",
        f"Translation: {best['translation']}",
        f"Meaning: {best['meaning']}",
    ]
    if len(candidates) > 1:
        lines.append("\nOther close matches:")
        for candidate in candidates[1:3]:
            lines.append(f"- {candidate['idiom']} -> {candidate['translation']}")
    return "\n".join(lines)


def build_chat_fn(config: ProjectConfig):
    def chat_fn(message: str, history: list[dict]) -> str:
        candidates = retrieve_candidates(message, config, top_k=config.chat_top_k)

        if config.can_use_chat_api():
            try:
                return _call_chat_api(message, history, candidates, config)
            except Exception as error:
                print(f"Chat API fallback triggered: {error}")
                return _fallback_response(message, candidates)

        return _fallback_response(message, candidates)

    return chat_fn


def main() -> None:
    config = ProjectConfig()
    chat_fn = build_chat_fn(config)

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="Idiom Translator Chatbot",
        description="Type an idiom or sentence. Uses Qdrant retrieval + optional chat LLM API.",
    )

    demo.launch(server_name=config.gui_host, server_port=config.gui_port, inbrowser=True)


if __name__ == "__main__":
    main()
