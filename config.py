"""Global configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ── Qwen (DashScope) ──────────────────────────────────────────────
QWEN_API_KEY: str = _env("QWEN_API_KEY")
QWEN_MODEL: str = _env("QWEN_MODEL", "qwen-plus")
QWEN_BASE_URL: str = _env(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode"
)

# ── GLM (智谱 BigModel) ───────────────────────────────────────────
GLM_API_KEY: str = _env("GLM_API_KEY")
GLM_MODEL: str = _env("GLM_MODEL", "glm-4-flash")
GLM_BASE_URL: str = _env(
    "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas"
)

# ── DeepSeek ──────────────────────────────────────────────────────
DEEPSEEK_API_KEY: str = _env("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL: str = _env("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL: str = _env(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
)
