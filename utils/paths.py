from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent


def app_path(*parts) -> Path:
    if not parts:
        return APP_ROOT

    path = Path(parts[0])
    if not path.is_absolute():
        path = APP_ROOT / path

    if len(parts) > 1:
        path = path.joinpath(*parts[1:])
    return path


def app_str(*parts) -> str:
    return str(app_path(*parts))
