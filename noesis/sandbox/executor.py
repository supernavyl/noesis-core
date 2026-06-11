"""Docker sandbox executor — runs untrusted model-generated code in isolation."""

from __future__ import annotations

import asyncio
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


SUPPORTED_LANGS = ("python", "bash", "node", "rust")


class DockerExecutor:
    def __init__(
        self,
        image_python: str = "python:3.12-slim",
        image_node: str = "node:22-slim",
        image_rust: str = "rust:1.83-slim",
        memory_limit: str = "2g",
        cpu_limit: str = "1.0",
        network: Literal["none", "bridge"] = "none",
        default_timeout_s: int = 60,
    ) -> None:
        self.image_python = image_python
        self.image_node = image_node
        self.image_rust = image_rust
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network = network
        self.default_timeout_s = default_timeout_s

    async def run(
        self,
        code: str,
        lang: Literal["python", "bash", "node", "rust"] = "python",
        timeout_s: int | None = None,
        stdin: str | None = None,
    ) -> ExecResult:
        if lang not in SUPPORTED_LANGS:
            raise ValueError(f"Unsupported language: {lang}")
        timeout_s = timeout_s or self.default_timeout_s
        image, cmd, file_ext = self._image_and_cmd(lang)

        loop = asyncio.get_event_loop()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / f"prog{file_ext}"
            src.write_text(textwrap.dedent(code))

            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                self.network,
                "--memory",
                self.memory_limit,
                "--cpus",
                self.cpu_limit,
                "--read-only",
                "--tmpfs",
                "/tmp:rw,size=128m",
                "-v",
                f"{src}:/work/prog{file_ext}:ro",
                "-w",
                "/work",
                image,
                *cmd,
            ]

            start = loop.time()
            try:
                proc = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdin=asyncio.subprocess.PIPE if stdin else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin.encode() if stdin else None), timeout=timeout_s
                )
                return ExecResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    timed_out=False,
                    duration_s=loop.time() - start,
                )
            except TimeoutError:
                # Caller responsible for cleaning up the container — rely on --rm + SIGKILL.
                return ExecResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"timed out after {timeout_s}s",
                    timed_out=True,
                    duration_s=loop.time() - start,
                )

    def _image_and_cmd(self, lang: str) -> tuple[str, list[str], str]:
        if lang == "python":
            return self.image_python, ["python", "/work/prog.py"], ".py"
        if lang == "bash":
            return "bash:5", ["bash", "/work/prog.sh"], ".sh"
        if lang == "node":
            return self.image_node, ["node", "/work/prog.js"], ".js"
        if lang == "rust":
            return (
                self.image_rust,
                ["sh", "-c", "rustc /work/prog.rs -o /tmp/prog && /tmp/prog"],
                ".rs",
            )
        raise ValueError(lang)
