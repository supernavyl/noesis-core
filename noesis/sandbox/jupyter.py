"""Persistent Jupyter kernel session — maintains state across exec calls.

Used when the model wants to build up a multi-step experiment incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jupyter_client.manager import KernelManager


@dataclass
class CellResult:
    stdout: str
    stderr: str
    result: Any
    error: str | None


class JupyterSession:
    """Wraps a local Python kernel. Keep simple — heavy isolation lives in DockerExecutor."""

    def __init__(self, kernel_name: str = "python3") -> None:
        self.km = KernelManager(kernel_name=kernel_name)
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=30)

    def execute(self, code: str, timeout: float = 60.0) -> CellResult:
        msg_id = self.kc.execute(code)
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        result: Any = None
        error: str | None = None

        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=timeout)
            except Exception:
                error = "timeout"
                break
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            mtype = msg["msg_type"]
            content = msg["content"]
            if mtype == "stream":
                if content["name"] == "stdout":
                    stdout_chunks.append(content["text"])
                else:
                    stderr_chunks.append(content["text"])
            elif mtype == "execute_result":
                result = content["data"].get("text/plain")
            elif mtype == "error":
                error = "\n".join(content.get("traceback", []))
            elif mtype == "status" and content["execution_state"] == "idle":
                break

        return CellResult(
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            result=result,
            error=error,
        )

    def shutdown(self) -> None:
        try:
            self.kc.stop_channels()
        finally:
            self.km.shutdown_kernel(now=True)
