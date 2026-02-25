#!/usr/bin/env python3
"""
CodeForge — Autonomous Repo Builder
Provider-agnostic agentic framework. Supports Anthropic, OpenAI, OpenRouter,
Google Gemini, Moonshot AI, Qwen, Mistral, Groq, Together AI, Ollama, and more.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from codeforge.config import ForgeConfig
from codeforge.display import ForgeDisplay
from codeforge.onboarding import run_onboarding, run_config_menu


def parse_args():
    parser = argparse.ArgumentParser(
        prog="codeforge",
        description="Autonomous code repo builder — any AI provider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  codeforge build "a REST API for a todo app using FastAPI and SQLite"
  codeforge build "a TypeScript Discord bot" --lang typescript --out ./my-bot
  codeforge build "a Go gRPC auth service" --model gpt-4o --max-cycles 40
  codeforge resume ./my-project         # resume an interrupted session
  codeforge inspect ./my-project        # inspect memory and plan state
  codeforge test ./my-project           # test + self-critique pass
  codeforge config                      # reconfigure providers / settings
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    build_p = subparsers.add_parser("build", help="Start a new autonomous repo build")
    build_p.add_argument("prompt", type=str, help="What to build (natural language)")
    build_p.add_argument("--lang", type=str, default=None, help="Primary language hint (python, ts, go, rust, etc.)")
    build_p.add_argument("--out", type=str, default=None, help="Output directory")
    build_p.add_argument("--provider", type=str, default=None, help="Override active provider (e.g. openrouter)")
    build_p.add_argument("--model", type=str, default=None, help="Override model for this run")
    build_p.add_argument("--max-cycles", type=int, default=None, help="Max agent loop cycles")
    build_p.add_argument("--verbose", "-v", action="store_true", default=None)
    build_p.add_argument("--dry-run", action="store_true", help="Plan only, do not write files")

    # resume
    resume_p = subparsers.add_parser("resume", help="Resume an interrupted forge session")
    resume_p.add_argument("path", type=str)
    resume_p.add_argument("--provider", type=str, default=None)
    resume_p.add_argument("--model", type=str, default=None)
    resume_p.add_argument("--max-cycles", type=int, default=None)
    resume_p.add_argument("--verbose", "-v", action="store_true", default=None)

    # inspect
    inspect_p = subparsers.add_parser("inspect", help="Inspect forge memory and plan state")
    inspect_p.add_argument("path", type=str)

    # test
    test_p = subparsers.add_parser("test", help="Run test + self-critique pass")
    test_p.add_argument("path", type=str)
    test_p.add_argument("--verbose", "-v", action="store_true", default=None)

    # config
    subparsers.add_parser("config", help="Configure providers, API keys, and settings")

    # providers
    subparsers.add_parser("providers", help="List all available providers and their status")

    return parser.parse_args()


def _resolve_config_with_overrides(config: ForgeConfig, args) -> ForgeConfig:
    """Apply any CLI override flags (--provider, --model) to a copy of config."""
    provider_id = getattr(args, "provider", None)
    model_override = getattr(args, "model", None)

    if provider_id:
        if provider_id not in config.providers:
            print(f"\n  Provider '{provider_id}' not configured. Run `codeforge config` to add it.\n")
            sys.exit(1)
        config.active_provider = provider_id

    if model_override:
        active = config.get_active()
        if active:
            from copy import deepcopy
            config = deepcopy(config)
            config.providers[config.active_provider].model = model_override

    return config


def _print_providers(config: ForgeConfig):
    from codeforge.providers import PROVIDER_REGISTRY
    from codeforge.display import _c, BOLD, GREEN, RED, CYAN, DIM, YELLOW

    print()
    print(_c("  Available providers:", BOLD))
    print()
    for pid, meta in PROVIDER_REGISTRY.items():
        pcfg = config.providers.get(pid)
        if pcfg and pcfg.is_ready():
            status = _c("✓ configured", GREEN)
            model_str = _c(f"  ({pcfg.model})", DIM)
        else:
            status = _c("○ not set up", DIM)
            model_str = ""

        active_mark = _c("  ← active", CYAN) if pid == config.active_provider else ""
        print(f"  {status}  {_c(meta.name, BOLD)}{model_str}{active_mark}")
        print(_c(f"             {meta.description}", DIM))
        if meta.docs_url:
            print(_c(f"             {meta.docs_url}", DIM))
        print()


async def main():
    args = parse_args()
    display = ForgeDisplay()
    config = ForgeConfig.load()

    # ── config command ────────────────────────────────────────────────────
    if args.command == "config":
        if config.onboarding_complete and config.providers:
            run_config_menu(config, display)
        else:
            run_onboarding(config, display)
        return

    # ── providers command ─────────────────────────────────────────────────
    if args.command == "providers":
        _print_providers(config)
        return

    # ── First-run gate: onboarding ────────────────────────────────────────
    if not config.onboarding_complete or not config.get_active() or not config.get_active().is_ready():
        from codeforge.display import _c, CYAN, BOLD
        print()
        print(_c("  Welcome to CodeForge! Let's get you set up first.", BOLD))
        config = run_onboarding(config, display)
        print()

    # ── inspect ───────────────────────────────────────────────────────────
    if args.command == "inspect":
        from codeforge.memory import ForgeMemory
        forge_dir = Path(args.path) / ".forge"
        if not forge_dir.exists():
            display.error(f"No .forge/ state found in {args.path}")
            return
        mem = ForgeMemory.load(forge_dir)
        display.print_inspection(mem)
        return

    # ── build / resume / test — need a live provider ──────────────────────
    config = _resolve_config_with_overrides(config, args)

    from codeforge.forge import CodeForge

    verbose = getattr(args, "verbose", None)

    if args.command == "build":
        forge = CodeForge(
            config=config,
            max_cycles=args.max_cycles,
            verbose=verbose if verbose else None,
            dry_run=args.dry_run,
            display=display,
        )
        out_path = Path(args.out) if args.out else None
        await forge.build(prompt=args.prompt, lang_hint=args.lang, out_dir=out_path)

    elif args.command == "resume":
        forge = CodeForge(
            config=config,
            max_cycles=args.max_cycles,
            verbose=verbose if verbose else None,
            display=display,
        )
        await forge.resume(Path(args.path))

    elif args.command == "test":
        forge = CodeForge(
            config=config,
            max_cycles=10,
            verbose=verbose if verbose else None,
            display=display,
        )
        await forge.test_pass(Path(args.path))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n  [codeforge] Interrupted. State saved — run `codeforge resume` to continue.\n")
        sys.exit(0)
