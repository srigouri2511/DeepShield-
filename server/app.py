from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn
from openenv.core.env_server.http_server import create_app

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.models import DeepShieldAction, DeepShieldObservation
from server.openenv_adapter import DeepShieldOpenEnv


_SERVER_ENV = DeepShieldOpenEnv()


def get_env() -> DeepShieldOpenEnv:
    return _SERVER_ENV


app = create_app(
    get_env,
    DeepShieldAction,
    DeepShieldObservation,
    env_name="deepshield",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    if port is None:
        port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
