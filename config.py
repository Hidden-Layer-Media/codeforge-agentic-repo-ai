"""
config.py — User configuration for CodeForge.

Stored at ~/.config/codeforge/config.json
Each provider's API key is kept in its own slot.
Users can configure multiple providers and switch between them.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


CONFIG_DIR  = Path.home() / ".config" / "codeforge"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class ProviderConfig:
    provider_id: str
    api_key: str = ""
    model: str = ""
    base_url: str = ""      # for Ollama, custom endpoints

    def is_ready(self) -> bool:
        """True if this provider has enough config to actually be used."""
        from .providers import PROVIDER_REGISTRY
        meta = PROVIDER_REGISTRY.get(self.provider_id)
        if not meta:
            return False
        needs_key = self.provider_id not in ("ollama",)
        if needs_key and not self.api_key:
            return False
        if meta.needs_base_url and not self.base_url:
            return False
        return bool(self.model)


@dataclass
class ForgeConfig:
    # Active provider
    active_provider: str = ""

    # Per-provider configs keyed by provider_id
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # Global defaults
    default_max_cycles: int = 30
    default_max_tokens: int = 8192

    # UX
    verbose_by_default: bool = False
    onboarding_complete: bool = False

    @classmethod
    def load(cls) -> "ForgeConfig":
        if not CONFIG_FILE.exists():
            return cls()
        try:
            data = json.loads(CONFIG_FILE.read_text())
            providers = {
                k: ProviderConfig(**v)
                for k, v in data.get("providers", {}).items()
            }
            return cls(
                active_provider=data.get("active_provider", ""),
                providers=providers,
                default_max_cycles=data.get("default_max_cycles", 30),
                default_max_tokens=data.get("default_max_tokens", 8192),
                verbose_by_default=data.get("verbose_by_default", False),
                onboarding_complete=data.get("onboarding_complete", False),
            )
        except Exception:
            return cls()

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "active_provider": self.active_provider,
            "providers": {k: asdict(v) for k, v in self.providers.items()},
            "default_max_cycles": self.default_max_cycles,
            "default_max_tokens": self.default_max_tokens,
            "verbose_by_default": self.verbose_by_default,
            "onboarding_complete": self.onboarding_complete,
        }
        CONFIG_FILE.write_text(json.dumps(data, indent=2))

    def get_active(self) -> Optional[ProviderConfig]:
        return self.providers.get(self.active_provider)

    def set_provider(self, provider_id: str, api_key: str = "", model: str = "", base_url: str = ""):
        from .providers import PROVIDER_REGISTRY
        meta = PROVIDER_REGISTRY.get(provider_id)
        if meta and not model:
            model = meta.default_model
        self.providers[provider_id] = ProviderConfig(
            provider_id=provider_id,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )

    def build_provider_instance(self):
        """Build and return the active provider object, ready to call."""
        from .providers import build_provider
        cfg = self.get_active()
        if not cfg:
            raise RuntimeError(
                "No active provider configured. Run `codeforge config` to set one up."
            )
        if not cfg.is_ready():
            raise RuntimeError(
                f"Provider '{cfg.provider_id}' is not fully configured. Run `codeforge config`."
            )
        return build_provider(
            provider_id=cfg.provider_id,
            api_key=cfg.api_key,
            model=cfg.model,
            base_url=cfg.base_url or None,
        )

    def summary_line(self) -> str:
        cfg = self.get_active()
        if not cfg:
            return "No provider configured"
        from .providers import PROVIDER_REGISTRY
        meta = PROVIDER_REGISTRY.get(cfg.provider_id)
        name = meta.name if meta else cfg.provider_id
        return f"{name} / {cfg.model}"
