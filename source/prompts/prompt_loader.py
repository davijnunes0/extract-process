from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(task: str, variant: str) -> str:
    prompt_path = PROMPTS_DIR / task / f"{variant}.txt"

    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Prompt not found for task={task!r}, variant={variant!r}: {prompt_path}"
        )

    return prompt_path.read_text(encoding="utf-8")
