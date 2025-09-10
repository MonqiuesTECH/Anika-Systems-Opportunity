import math
import tiktoken

def fmt_ms(ms: int) -> str:
    return f"{ms:,} ms" if ms < 1000 else f"{ms/1000:.2f} s"

class TokenCounter:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # model name only affects encoding choice; safe default
        try:
            self.enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.enc = tiktoken.get_encoding("cl100k_base")

    def estimate_prompt_tokens(self, text: str) -> int:
        try:
            return len(self.enc.encode(text or ""))
        except Exception:
            # rough fallback
            return max(1, math.ceil(len(text or "") / 4))
