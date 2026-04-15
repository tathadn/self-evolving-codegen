"""Shared base class for pipeline agents.

Every agent in the code-generation pipeline constructs a ``ChatAnthropic`` LLM
and loads a system prompt from ``prompts/``. ``BaseAgent`` centralises that
plumbing so concrete agents only declare model, token budget, and prompt name.
``TesterBaseAgent`` extends it with the generation-aware fallback chain used
by the self-evolving tester.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from langchain_anthropic import ChatAnthropic

from config import MAX_TOKENS

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class BaseAgent:
    """Common LLM + system-prompt wiring for pipeline agents.

    Subclasses declare three class-level attributes and inherit the rest:

    - ``model_name``: Anthropic model id (usually sourced from ``config.py``).
    - ``max_tokens_key``: key into ``config.MAX_TOKENS`` for this agent's cap.
    - ``prompt_name``: basename (without ``.md``) of the file in ``prompts/``.
    """

    model_name: ClassVar[str]
    max_tokens_key: ClassVar[str]
    prompt_name: ClassVar[str]

    def __init__(self) -> None:
        self.llm: ChatAnthropic = ChatAnthropic(  # type: ignore[call-arg]
            model=self.model_name,
            max_tokens=MAX_TOKENS[self.max_tokens_key],
        )
        self.system_prompt: str = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load ``prompts/{prompt_name}.md`` as the system prompt."""
        return (_PROMPTS_DIR / f"{self.prompt_name}.md").read_text()


class TesterBaseAgent(BaseAgent):
    """Tester agent with generation-aware prompt resolution.

    Generation 0 uses the original ``prompts/tester.md``. Generation N > 0
    uses ``prompts/tester_gen_{N}.txt``, falling back to the nearest earlier
    generation that exists, then to the base prompt.
    """

    prompt_name: ClassVar[str] = "tester"
    generation: int

    def __init__(self, generation: int = 0) -> None:
        self.generation = generation
        super().__init__()

    def _load_prompt(self) -> str:
        if self.generation == 0:
            return (_PROMPTS_DIR / "tester.md").read_text()
        for gen in range(self.generation, 0, -1):
            path = _PROMPTS_DIR / f"tester_gen_{gen}.txt"
            if path.exists():
                return path.read_text()
        return (_PROMPTS_DIR / "tester.md").read_text()
