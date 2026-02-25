"""
onboarding.py — Interactive first-run setup wizard for CodeForge.

Runs on first `codeforge build` (or `codeforge config`) if no config exists.
Walks the user through picking a provider, entering their API key, and
choosing a model. No external deps — pure stdlib input().
"""

from __future__ import annotations
import sys
import os
from typing import Optional

from .config import ForgeConfig, ProviderConfig
from .providers import PROVIDER_REGISTRY, ProviderMeta
from .display import (
    ForgeDisplay, RESET, BOLD, DIM, RED, GREEN, YELLOW, BLUE,
    CYAN, MAGENTA, WHITE, _c
)


def _print(text: str = ""):
    print(text)


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else default


def _ask_secret(prompt: str) -> str:
    """Like _ask but masks input if possible."""
    try:
        import getpass
        val = getpass.getpass(f"  {prompt}: ").strip()
    except Exception:
        val = _ask(prompt)
    return val


def _choose(prompt: str, options: list[tuple[str, str]], default: int = 1) -> str:
    """
    Display a numbered menu and return the chosen key.
    options: list of (key, label)
    """
    _print()
    _print(_c(f"  {prompt}", BOLD))
    for i, (key, label) in enumerate(options, 1):
        marker = _c("›", CYAN) if i == default else " "
        _print(f"  {marker} {_c(str(i), CYAN + BOLD)}. {label}")
    _print()

    while True:
        raw = _ask(f"Enter number (default {default})", str(default))
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        except ValueError:
            pass
        _print(_c("  Please enter a valid number.", RED))


def _header(text: str):
    width = 60
    _print()
    _print(_c("  " + "─" * width, CYAN))
    _print(_c(f"  {text}", BOLD + CYAN))
    _print(_c("  " + "─" * width, CYAN))
    _print()


def _section(text: str):
    _print()
    _print(_c(f"  ◆ {text}", BOLD))


def _note(text: str):
    _print(_c(f"    {text}", DIM))


def _ok(text: str):
    _print(_c(f"  ✓ {text}", GREEN))


def _warn(text: str):
    _print(_c(f"  ⚠  {text}", YELLOW))


def run_onboarding(config: ForgeConfig, display: ForgeDisplay, reconfigure: bool = False) -> ForgeConfig:
    """
    Full interactive onboarding. Returns the updated config (also saved to disk).
    """
    _print()
    _print(_c("  ╔══════════════════════════════════════════════════════════╗", CYAN + BOLD))
    _print(_c("  ║             WELCOME TO  C O D E F O R G E  🔥            ║", CYAN + BOLD))
    _print(_c("  ║         Autonomous repo builder — any AI provider         ║", CYAN))
    _print(_c("  ╚══════════════════════════════════════════════════════════╝", CYAN + BOLD))
    _print()
    _print(_c("  Let's get you set up in about 60 seconds.", DIM))
    _print(_c("  You can re-run this wizard anytime with: codeforge config", DIM))
    _print()

    input(_c("  Press Enter to begin...", DIM))

    # ── STEP 1: Choose provider ───────────────────────────────────────────
    _header("STEP 1 of 3 — Choose your AI provider")

    _print(_c("  CodeForge works with any of these providers:", DIM))
    _print()

    # Build ordered menu
    provider_order = [
        "anthropic", "openai", "openrouter", "gemini",
        "moonshot", "qwen", "mistral", "groq", "together", "ollama", "custom"
    ]
    options = []
    for pid in provider_order:
        meta = PROVIDER_REGISTRY[pid]
        suggested = f"  {_c('(e.g. ' + meta.default_model + ')', DIM)}" if meta.default_model else ""
        label = f"{_c(meta.name, BOLD)}  —  {meta.description}"
        options.append((pid, label))

    # Highlight if already configured
    if config.active_provider:
        _note(f"Currently active: {config.summary_line()}")

    chosen_pid = _choose("Which provider would you like to use?", options, default=1)
    meta = PROVIDER_REGISTRY[chosen_pid]

    _print()
    _ok(f"Selected: {meta.name}")
    if meta.docs_url:
        _note(f"API keys / docs: {meta.docs_url}")

    # ── STEP 2: API key ───────────────────────────────────────────────────
    _header("STEP 2 of 3 — API credentials")

    existing_cfg = config.providers.get(chosen_pid)
    needs_key = chosen_pid != "ollama"

    api_key = ""
    base_url = ""

    if needs_key:
        # Check env var first
        env_hints = {
            "anthropic":  "ANTHROPIC_API_KEY",
            "openai":     "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "gemini":     "GOOGLE_API_KEY",
            "moonshot":   "MOONSHOT_API_KEY",
            "qwen":       "DASHSCOPE_API_KEY",
            "mistral":    "MISTRAL_API_KEY",
            "groq":       "GROQ_API_KEY",
            "together":   "TOGETHER_API_KEY",
            "custom":     "CUSTOM_API_KEY",
        }
        env_var = env_hints.get(chosen_pid, "")
        env_val = os.environ.get(env_var, "") if env_var else ""

        if env_val:
            _ok(f"Found {env_var} in environment — using it.")
            _note("(You can override by entering a new key below, or just press Enter)")
            api_key = _ask_secret(f"API key (or Enter to use env)") or env_val
        elif existing_cfg and existing_cfg.api_key:
            masked = existing_cfg.api_key[:6] + "..." + existing_cfg.api_key[-4:]
            _note(f"Existing key on file: {masked}")
            new_key = _ask_secret("New API key (or Enter to keep existing)")
            api_key = new_key or existing_cfg.api_key
        else:
            _print(_c(f"  Paste your {meta.name} API key below.", DIM))
            _note("It will be stored in ~/.config/codeforge/config.json (not shared)")
            while not api_key:
                api_key = _ask_secret("API key")
                if not api_key:
                    _warn("API key cannot be empty.")

    if meta.needs_base_url or chosen_pid == "ollama":
        existing_url = existing_cfg.base_url if existing_cfg else ""
        default_url = existing_url or meta.base_url_hint
        _print()
        _print(_c(f"  Base URL for {meta.name}:", DIM))
        base_url = _ask("Base URL", default_url)

    _ok("Credentials saved.")

    # ── STEP 3: Choose model ──────────────────────────────────────────────
    _header("STEP 3 of 3 — Choose a model")

    suggested = meta.suggested_models
    model = ""

    if suggested:
        _print(_c(f"  Suggested models for {meta.name}:", DIM))
        model_options = [(m, m) for m in suggested] + [("custom", _c("Enter a different model name...", DIM))]
        chosen_model = _choose("Which model?", model_options, default=1)
        if chosen_model == "custom":
            model = _ask("Model name", meta.default_model)
        else:
            model = chosen_model
    else:
        model = _ask("Model name", meta.default_model or "")

    if not model:
        model = meta.default_model or "gpt-4o"

    _ok(f"Model: {model}")

    # ── Save ──────────────────────────────────────────────────────────────
    config.set_provider(chosen_pid, api_key=api_key, model=model, base_url=base_url)
    config.active_provider = chosen_pid
    config.onboarding_complete = True
    config.save()

    # ── Optional: configure more ──────────────────────────────────────────
    _print()
    _print(_c("  ── Optional settings ──────────────────────────────────────", DIM))
    _print()

    raw_cycles = _ask("Default max cycles per build", str(config.default_max_cycles))
    try:
        config.default_max_cycles = max(5, int(raw_cycles))
    except ValueError:
        pass

    verbose_input = _ask("Verbose tool output by default? (y/n)", "n").lower()
    config.verbose_by_default = verbose_input in ("y", "yes")

    config.save()

    # ── Done ──────────────────────────────────────────────────────────────
    _print()
    _print(_c("  ╔══════════════════════════════════════════════════════════╗", GREEN + BOLD))
    _print(_c("  ║                  Setup complete! 🔥                       ║", GREEN + BOLD))
    _print(_c("  ╚══════════════════════════════════════════════════════════╝", GREEN + BOLD))
    _print()
    _print(f"  {_c('Active provider:', BOLD)} {config.summary_line()}")
    _print(f"  {_c('Config file:', BOLD)}    {_c('~/.config/codeforge/config.json', DIM)}")
    _print()
    _print(_c("  You're ready. Try:", BOLD))
    _print()
    _print(_c('    codeforge build "a FastAPI todo API with SQLite"', CYAN))
    _print(_c('    codeforge build "a TypeScript Discord bot"', CYAN))
    _print(_c('    codeforge build "a Rust CLI markdown-to-HTML converter"', CYAN))
    _print()

    return config


def run_config_menu(config: ForgeConfig, display: ForgeDisplay) -> ForgeConfig:
    """
    Non-first-run config menu — list current state, offer to change things.
    """
    _header("CODEFORGE CONFIG")

    _print(f"  {_c('Active provider:', BOLD)} {config.summary_line()}")
    _print(f"  {_c('Config file:', BOLD)} {CONFIG_FILE_PATH()}")
    _print(f"  {_c('Max cycles:', BOLD)} {config.default_max_cycles}")
    _print(f"  {_c('Verbose:', BOLD)} {config.verbose_by_default}")
    _print()

    if config.providers:
        _print(_c("  Configured providers:", BOLD))
        for pid, pcfg in config.providers.items():
            meta = PROVIDER_REGISTRY.get(pid)
            name = meta.name if meta else pid
            status = _c("✓ ready", GREEN) if pcfg.is_ready() else _c("✗ incomplete", RED)
            active_mark = _c(" ← active", CYAN) if pid == config.active_provider else ""
            masked_key = (pcfg.api_key[:4] + "..." if pcfg.api_key else "no key")
            _print(f"    {status}  {_c(name, BOLD)} ({pcfg.model})  {masked_key}{active_mark}")
        _print()

    options = [
        ("reconfigure", "Add or change a provider"),
        ("switch",      "Switch active provider"),
        ("defaults",    "Change default settings"),
        ("quit",        "Back / quit"),
    ]
    choice = _choose("What would you like to do?", options, default=1)

    if choice == "reconfigure":
        return run_onboarding(config, display, reconfigure=True)

    elif choice == "switch":
        ready = [(pid, f"{PROVIDER_REGISTRY.get(pid, pid).name if isinstance(PROVIDER_REGISTRY.get(pid), ProviderMeta) else pid} ({pcfg.model})")
                 for pid, pcfg in config.providers.items() if pcfg.is_ready()]
        if not ready:
            _warn("No ready providers configured yet. Run setup first.")
            return run_onboarding(config, display)
        chosen = _choose("Switch to which provider?", ready)
        config.active_provider = chosen
        config.save()
        _ok(f"Active provider set to: {config.summary_line()}")

    elif choice == "defaults":
        raw = _ask("Default max cycles", str(config.default_max_cycles))
        try:
            config.default_max_cycles = max(5, int(raw))
        except ValueError:
            pass
        v = _ask("Verbose by default? (y/n)", "y" if config.verbose_by_default else "n").lower()
        config.verbose_by_default = v in ("y", "yes")
        config.save()
        _ok("Settings saved.")

    return config


def CONFIG_FILE_PATH() -> str:
    from .config import CONFIG_FILE
    return str(CONFIG_FILE).replace(str(Path.home()), "~")


from pathlib import Path
