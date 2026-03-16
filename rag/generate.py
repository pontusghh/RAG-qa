import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def extract_yes_no(raw: str) -> str | None:
    """Extract normalized yes/no from model output."""
    text = raw.strip().lower()
    if not text:
        return None

    first = text.split()[0]
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"

    match = re.search(r"\b(yes|no)\b", text)
    if match:
        return match.group(1)
    return None


def _build_llm(config: dict) -> ChatOpenAI:
    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("provider", "groq").lower()

    if provider != "groq":
        raise ValueError("Only 'groq' provider is supported in this project.")

    api_key_env = llm_cfg.get("api_key_env", "GROQ_API_KEY")
    api_base = llm_cfg.get("api_base", "https://api.groq.com/openai/v1")
    model_name = llm_cfg.get("model_name", "llama-3.1-8b-instant")
    api_key = os.getenv(api_key_env)

    if not api_key:
        raise ValueError(
            f"Missing API key in environment variable '{api_key_env}'. "
            "Set it in your shell or .env before calling /ask."
        )

    model_kwargs = {
        "model": model_name,
        "temperature": llm_cfg.get("temperature", 0.0),
        "max_tokens": llm_cfg.get("max_tokens", 30),
        "api_key": api_key,
    }
    if api_base:
        model_kwargs["base_url"] = api_base

    return ChatOpenAI(**model_kwargs)


def generate_answer(question: str, context: str, config: dict):
    """Generate answer using API-based LLM."""
    model = _build_llm(config)

    prompt = (
        "You are a biomedical expert.\n"
        "Answer the question using the context.\n"
        "Reply with only one word: yes or no.\n"
        "Do not explain.\n\n"

        "Example:\n"
        "Question: Do mitochondria play a role in programmed cell death?\n"
        "Context: Mitochondrial dynamics were observed during programmed cell death.\n"
        "Answer: yes\n\n"

        "Example:\n"
        "Question: Does deprivation have no effect on mortality?\n"
        "Context: Deprivation is associated with higher mortality.\n"
        "Answer: no\n\n"

        "Question: " + question + "\n"
        "Context: " + context + "\n"
        "Answer:"
    )

    response = model.invoke(prompt)
    raw_answer = response.content if isinstance(response.content, str) else str(response.content)
    pred = extract_yes_no(raw_answer)
    return pred, raw_answer


def generate_lm_only_answer(question: str, config: dict):
    """Generate yes/no answer without retrieved context."""
    model = _build_llm(config)

    prompt = (
        "You are a biomedical expert. Answer the following yes/no question "
        "using your knowledge. Output ONLY 'yes' or 'no'.\n\n"
        "Example:\n"
        "Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\n"
        "Answer: yes\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    response = model.invoke(prompt)
    raw_answer = response.content if isinstance(response.content, str) else str(response.content)
    pred = extract_yes_no(raw_answer)
    return pred, raw_answer
