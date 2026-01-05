from typing import Any


def truncate_strings(data: Any, max_length: int = 100) -> Any:
    if isinstance(data, dict):
        return {k: truncate_strings(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_strings(item, max_length) for item in data]
    elif isinstance(data, str):
        return data if len(data) <= max_length else data[:max_length] + "..."
    else:
        return data
