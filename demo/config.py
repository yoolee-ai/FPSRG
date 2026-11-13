"""Environment-only deployment configuration; no site-specific paths."""
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    home: Path
    api_url: str
    api_token: str
    deepseek_url: str
    deepseek_key: str
    deepseek_model: str
    semantic_support: bool
    max_tokens: int
    timeout: float

    @classmethod
    def from_env(cls):
        return cls(
            home=Path(os.getenv('FPSRG_HOME', './var/fpsrg')).expanduser().resolve(),
            api_url=os.getenv('FPSRG_API_URL', 'http://127.0.0.1:8000').rstrip('/'),
            api_token=os.getenv('FPSRG_API_TOKEN', ''),
            deepseek_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com').rstrip('/'),
            deepseek_key=os.getenv('DEEPSEEK_API_KEY', ''),
            deepseek_model=os.getenv('DEEPSEEK_MODEL', 'deepseek-flash'),
            semantic_support=os.getenv('FPSRG_SEMANTIC_SUPPORT', 'true').lower() == 'true',
            max_tokens=int(os.getenv('FPSRG_MAX_TOKENS', '4096')),
            timeout=float(os.getenv('FPSRG_REQUEST_TIMEOUT', '900')),
        )
