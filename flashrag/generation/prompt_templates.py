"""
RAG prompt templates for context-augmented generation.

Each template formats retrieved contexts and the user query into a prompt
suitable for an LLM to generate a grounded answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PromptTemplate:
    """
    A prompt template with ``{context}`` and ``{question}`` placeholders.

    Attributes
    ----------
    name : str
        Template identifier.
    system : str
        System-level instruction.
    user : str
        User-turn template with ``{context}`` and ``{question}`` slots.
    context_separator : str
        How individual context passages are joined.
    context_prefix : str
        Label prepended to each context passage.
    """

    name: str
    system: str
    user: str
    context_separator: str = "\n\n"
    context_prefix: str = "[{idx}] "

    def format(
        self,
        question: str,
        contexts: List[str],
        sources: Optional[List[str]] = None,
    ) -> str:
        """Render the full prompt with contexts and question."""
        numbered = []
        for i, ctx in enumerate(contexts):
            prefix = self.context_prefix.format(idx=i + 1)
            source_tag = ""
            if sources and i < len(sources):
                source_tag = f" (source: {sources[i]})"
            numbered.append(f"{prefix}{ctx}{source_tag}")

        context_block = self.context_separator.join(numbered)
        user_text = self.user.format(context=context_block, question=question)

        if self.system:
            return f"{self.system}\n\n{user_text}"
        return user_text

    def format_messages(
        self,
        question: str,
        contexts: List[str],
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Return chat-style messages list for chat-format models."""
        numbered = []
        for i, ctx in enumerate(contexts):
            prefix = self.context_prefix.format(idx=i + 1)
            source_tag = ""
            if sources and i < len(sources):
                source_tag = f" (source: {sources[i]})"
            numbered.append(f"{prefix}{ctx}{source_tag}")

        context_block = self.context_separator.join(numbered)
        user_text = self.user.format(context=context_block, question=question)

        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": user_text})
        return messages


_TEMPLATES: Dict[str, PromptTemplate] = {
    "default": PromptTemplate(
        name="default",
        system=(
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the context does not contain enough information, say so honestly. "
            "Cite the source numbers [1], [2], etc. when using information from a specific context."
        ),
        user=(
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    ),
    "conversational": PromptTemplate(
        name="conversational",
        system=(
            "You are a friendly and knowledgeable assistant. Use the provided reference "
            "material to answer the user's question in a natural, conversational tone. "
            "If the references don't cover the topic, let the user know."
        ),
        user=(
            "Here are some relevant references:\n\n{context}\n\n"
            "User: {question}\n\n"
            "Assistant:"
        ),
    ),
    "academic": PromptTemplate(
        name="academic",
        system=(
            "You are an academic research assistant. Provide precise, well-structured "
            "answers grounded in the provided sources. Use in-text citations [1], [2] "
            "for every claim. Distinguish between what the sources state and your analysis."
        ),
        user=(
            "Sources:\n{context}\n\n"
            "Research Question: {question}\n\n"
            "Analysis:"
        ),
    ),
    "code": PromptTemplate(
        name="code",
        system=(
            "You are a programming assistant. Use the provided documentation and code "
            "snippets to answer the question. Include code examples when relevant. "
            "Reference the source documents by number."
        ),
        user=(
            "Documentation:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    ),
    "minimal": PromptTemplate(
        name="minimal",
        system="",
        user=(
            "Given the following information:\n{context}\n\n"
            "Answer this question: {question}"
        ),
    ),
}


def get_template(name: str = "default") -> PromptTemplate:
    """Get a prompt template by name."""
    if name not in _TEMPLATES:
        available = list(_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{name}'. Available: {available}")
    return _TEMPLATES[name]


def register_template(template: PromptTemplate) -> None:
    """Register a custom prompt template."""
    _TEMPLATES[template.name] = template


def list_templates() -> List[str]:
    return list(_TEMPLATES.keys())
