"""
forge.py — The CodeForge agent loop engine.

Provider-agnostic: works with Anthropic, OpenAI, OpenRouter, Gemini,
Moonshot, Qwen, Mistral, Groq, Together, Ollama, or any custom endpoint.

Architecture: PERCEIVE → PLAN → EXECUTE → MOLT (reflect + update memory) → repeat
"""

from __future__ import annotations
import json
import asyncio
from pathlib import Path
from typing import Optional

from .memory import ForgeMemory
from .tools import TOOL_DEFINITIONS, ToolDispatcher
from .display import ForgeDisplay
from .config import ForgeConfig
from .providers import BaseProvider, ForgeResponse
from .prompts import (
    SYSTEM_PROMPT,
    build_initial_prompt,
    build_resume_prompt,
    build_test_prompt,
)


DONE_SIGNAL = "__FORGE_COMPLETE__"


class CodeForge:
    def __init__(
        self,
        config: Optional[ForgeConfig] = None,
        provider: Optional[BaseProvider] = None,
        max_cycles: Optional[int] = None,
        verbose: Optional[bool] = None,
        dry_run: bool = False,
        display: Optional[ForgeDisplay] = None,
    ):
        self.display = display or ForgeDisplay()
        self._config = config or ForgeConfig.load()

        if provider:
            self._provider = provider
        else:
            self._provider = self._config.build_provider_instance()

        self.max_cycles = max_cycles if max_cycles is not None else self._config.default_max_cycles
        self.verbose = verbose if verbose is not None else self._config.verbose_by_default
        self.dry_run = dry_run

    @property
    def model(self) -> str:
        return self._provider.model

    @property
    def provider_name(self) -> str:
        return self._provider.provider_id

    async def build(self, prompt: str, lang_hint: Optional[str] = None, out_dir: Optional[Path] = None):
        self.display.banner("CODEFORGE 🔥", f"{self._config.summary_line()}  ·  {prompt[:60]}")

        memory = ForgeMemory.bootstrap(prompt=prompt, lang_hint=lang_hint)

        if out_dir is None:
            slug = memory.project_name.lower().replace(" ", "-")[:32]
            out_dir = Path(".") / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        memory.repo_path = out_dir
        memory.save(out_dir / ".forge")

        messages = [{"role": "user", "content": build_initial_prompt(prompt, lang_hint)}]
        await self._run_loop(memory, messages, out_dir)

    async def resume(self, repo_path: Path):
        forge_dir = repo_path / ".forge"
        if not forge_dir.exists():
            self.display.error(f"No .forge/ state found in {repo_path}")
            return

        memory = ForgeMemory.load(forge_dir)
        memory.repo_path = repo_path
        self.display.banner("CODEFORGE RESUME 🔥", f"{self._config.summary_line()}  ·  {memory.original_prompt[:60]}")

        messages = memory.message_history + [
            {"role": "user", "content": build_resume_prompt(memory)}
        ]
        await self._run_loop(memory, messages, repo_path)

    async def test_pass(self, repo_path: Path):
        forge_dir = repo_path / ".forge"
        memory = ForgeMemory.load(forge_dir) if forge_dir.exists() else ForgeMemory.bootstrap(prompt="unknown")
        memory.repo_path = repo_path
        self.display.banner("CODEFORGE TEST 🔥", f"Testing: {repo_path}")

        messages = [{"role": "user", "content": build_test_prompt(memory, repo_path)}]
        await self._run_loop(memory, messages, repo_path, test_mode=True)

    async def _run_loop(self, memory: ForgeMemory, messages: list, repo_path: Path, test_mode: bool = False):
        dispatcher = ToolDispatcher(repo_path=repo_path, memory=memory, dry_run=self.dry_run)
        cycle = 0

        while cycle < self.max_cycles:
            cycle += 1
            self.display.cycle(cycle, self.max_cycles, memory)

            system = SYSTEM_PROMPT + "\n\n" + memory.as_context_block()

            try:
                response: ForgeResponse = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._provider.complete(
                        system=system,
                        messages=messages,
                        tools=TOOL_DEFINITIONS,
                        max_tokens=self._config.default_max_tokens,
                    ),
                )
            except Exception as e:
                self.display.error(f"Provider error ({self.provider_name}): {e}")
                memory.save(repo_path / ".forge")
                break

            content_blocks = response.content_blocks
            stop_reason = response.stop_reason

            if self.verbose:
                self.display.token_usage(response)

            messages.append({"role": "assistant", "content": content_blocks})

            if stop_reason in ("end_turn", "stop"):
                full_text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
                if DONE_SIGNAL in full_text:
                    self.display.success("Forge complete!")
                    memory.status = "complete"
                    memory.save(repo_path / ".forge")
                    self.display.summary(memory, repo_path)
                    return

                self.display.warn("Agent stopped without tool calls. Nudging...")
                messages.append({
                    "role": "user",
                    "content": "Continue building. Use your tools. When fully done, call forge_complete.",
                })
                continue

            if stop_reason == "tool_use":
                tool_results = []

                for block in content_blocks:
                    if block.get("type") != "tool_use":
                        continue

                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_id = block["id"]

                    if self.verbose:
                        self.display.tool_call(tool_name, tool_input)

                    result = await dispatcher.dispatch(tool_name, tool_input)

                    if self.verbose:
                        self.display.tool_result(tool_name, result)
                    elif result.get("status") == "written":
                        self.display.file_written(result["path"], result.get("lines", 0))

                    if result.get("signal") == DONE_SIGNAL:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps(result),
                        })
                        messages.append({"role": "user", "content": tool_results})
                        memory.status = "complete"
                        memory.save(repo_path / ".forge")
                        self.display.success("Forge complete!")
                        self.display.summary(memory, repo_path)
                        return

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(result),
                    })

                messages.append({"role": "user", "content": tool_results})

            # MOLT phase
            memory.cycles_completed = cycle
            memory.file_tree = dispatcher.get_file_tree()
            memory.message_history = messages[-20:]
            memory.save(repo_path / ".forge")

        self.display.warn(f"Max cycles ({self.max_cycles}) reached. State saved.")
        memory.status = "incomplete"
        memory.save(repo_path / ".forge")
        self.display.hint("Run `codeforge resume <path>` to continue.")
