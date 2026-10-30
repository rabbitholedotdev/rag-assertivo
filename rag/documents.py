"""Minimal document type. Matches the LangChain Document surface used by retrieve/eval."""
from dataclasses import dataclass, field


@dataclass
class Chunk:
    page_content: str
    metadata: dict = field(default_factory=dict)
