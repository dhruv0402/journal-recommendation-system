import subprocess


def call_llm(prompt: str) -> str:
    """
    Call Ollama with a hard timeout.
    Never blocks forever.
    """

    try:
        process = subprocess.run(
            ["ollama", "run", "phi3:mini"],
            input=prompt + "\n",
            text=True,
            encoding="utf-8",
            errors="ignore",
            capture_output=True,
            timeout=20,          # ⬅️ HARD STOP after 20 seconds
            check=False,
        )
        return process.stdout.strip()

    except subprocess.TimeoutExpired:
        # LLM took too long → fail safely
        return ""
