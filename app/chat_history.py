def format_chat_history(history: list[dict], max_turns: int = 5) -> str:
    """Format the last *max_turns* user/assistant exchanges.

    Returns an empty string when *history* is empty.
    """
    if not history:
        return ""

    # Collect (user, assistant) pairs from the end
    pairs: list[tuple[str, str]] = []
    i = len(history) - 1
    while i >= 0 and len(pairs) < max_turns:
        if history[i]["role"] != "assistant":
            i -= 1
            continue
        assistant_content = history[i]["content"]
        user_content = ""
        if i - 1 >= 0 and history[i - 1]["role"] == "user":
            user_content = history[i - 1]["content"]
            i -= 2
        else:
            i -= 1
        pairs.append((user_content, assistant_content))

    pairs.reverse()
    if not pairs:
        return ""

    lines = ["Conversation History:\n"]
    for user_msg, assistant_msg in pairs:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
        lines.append("")
    return "\n".join(lines)
