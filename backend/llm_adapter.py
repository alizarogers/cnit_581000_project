"""
llm_adapter.py: thin wrapper around the Claude Messages API.

All model calls go through this class, so swapping the model (or provider)
only touches this file.
"""

from typing import Dict, List, Optional

import anthropic

from backend.config import config


class LLMAdapter:
    def __init__(self, client: Optional[anthropic.Anthropic] = None):
        if client is None:
            config.validate()
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._client = client
        self._temperature_supported = True

    @staticmethod
    def build_messages(history: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
        """Trim history to the configured window and append the new user turn.

        The Messages API requires the first message to come from the user and
        roles to alternate, so any leading assistant turns are dropped.
        """
        max_messages = config.MAX_HISTORY_TURNS * 2
        recent = [t for t in (history or []) if t.get("role") in ("user", "assistant")]
        recent = recent[-max_messages:]
        while recent and recent[0]["role"] != "user":
            recent.pop(0)
        return [*recent, {"role": "user", "content": user_message}]

    def generate(
        self,
        system_prompt: str,
        history: List[Dict[str, str]],
        user_message: str,
        temperature: Optional[float] = None,
    ) -> str:
        kwargs = dict(
            model=config.CLAUDE_MODEL,
            max_tokens=config.MAX_TOKENS,
            system=system_prompt,
            messages=self.build_messages(history, user_message),
        )
        if temperature is not None and self._temperature_supported:
            kwargs["temperature"] = temperature
        try:
            response = self._client.messages.create(**kwargs)
        except (TypeError, anthropic.BadRequestError) as e:
            # Newer SDK versions and models may not accept temperature
            # (TypeError from the SDK, 400 from the API); retry once without it.
            if "temperature" not in kwargs or "temperature" not in str(e).lower():
                raise
            self._temperature_supported = False
            kwargs.pop("temperature")
            response = self._client.messages.create(**kwargs)
        return "".join(block.text for block in response.content if block.type == "text")
