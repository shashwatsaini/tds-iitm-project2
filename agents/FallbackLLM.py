from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from typing import List, Any
import os
from dotenv import load_dotenv

load_dotenv()


class FallbackLLM(BaseChatModel):

    def __init__(self, primary_model="gemini-2.5-pro", fallback_model="gemini-2.5-flash", **kwargs):
        super().__init__(**kwargs)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment variables.")

        object.__setattr__(self, "primary_llm", ChatGoogleGenerativeAI(
            model=primary_model, api_key=api_key, max_retries=0
        ))
        object.__setattr__(self, "fallback_llm", ChatGoogleGenerativeAI(
            model=fallback_model, api_key=api_key, max_retries=0
        ))
        object.__setattr__(self, "use_fallback", False)

    def _generate(self, messages: List[BaseMessage], stop: List[str] = None, **kwargs: Any):
        if not self.use_fallback:
            try:
                return self.primary_llm._generate(messages, stop=stop, **kwargs)
            except ResourceExhausted:
                print("[FallbackLLM] Rate limit hit — switching to fallback model.")
                object.__setattr__(self, "use_fallback", True)
        return self.fallback_llm._generate(messages, stop=stop, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fallback-llm"
