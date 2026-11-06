"""Pluggable answer-generation LLM, selected via LLM_PROVIDER."""

from langchain_core.language_models import BaseLanguageModel

from docmind.config import settings


def get_llm() -> BaseLanguageModel:
    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0.2)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model, api_key=settings.anthropic_api_key, temperature=0.2
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline

        pipe = pipeline(
            "text2text-generation",
            model=settings.hf_llm_model,
            max_new_tokens=256,
        )
        return HuggingFacePipeline(pipeline=pipe)

    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")
