from __future__ import annotations

import os
from getpass import getpass
from pathlib import Path


ENV_PATH = Path(".env")


def load_env_file(env_path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def persist_env_values(values: dict[str, str], env_path: Path = ENV_PATH) -> None:
    existing = load_env_file(env_path)
    existing.update(values)
    body = "\n".join(f"{key}={value}" for key, value in sorted(existing.items()))
    env_path.write_text(body + "\n", encoding="utf-8")


def ensure_hf_credentials(prompt: bool = True, env_path: Path = ENV_PATH) -> tuple[str, str]:
    env_values = load_env_file(env_path)
    username = env_values.get("HF_USERNAME") or os.environ.get("HF_USERNAME", "")
    token = env_values.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")

    if prompt and not username:
        username = input("Hugging Face username: ").strip()
    if prompt and not token:
        token = getpass("Hugging Face token: ").strip()

    if not username or not token:
        raise RuntimeError("Missing Hugging Face credentials. Set HF_USERNAME and HF_TOKEN in .env or provide them when prompted.")

    persist_env_values({"HF_USERNAME": username, "HF_TOKEN": token}, env_path=env_path)
    os.environ["HF_USERNAME"] = username
    os.environ["HF_TOKEN"] = token
    return username, token
