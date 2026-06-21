"""
GraphRAG pipeline: Knowledge-graph-augmented retrieval.

Builds a knowledge graph from extracted entity–relation triples, then
combines graph traversal with vector similarity search to retrieve
contextually rich passages for answer generation.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from flashrag.data.chunkers import Chunk
from flashrag.pipelines.basic_rag import RAGResult
from flashrag.registry import PIPELINES

logger = logging.getLogger(__name__)


@PIPELINES.register("graph_rag")
class GraphRAGPipeline:
    """
    Knowledge-graph-based retrieval-augmented generation pipeline.

    Extracts entities and relations from indexed documents, builds an
    in-memory knowledge graph, and combines graph traversal with vector
    retrieval at query time.

    Parameters
    ----------
    embedding_model : str
        Sentence-transformer model name for vector embeddings.
    generator_model : str
        Generator model identifier for answer generation.
    chunk_size : int
        Document chunk size in characters.
    entity_extract_method : str
        Entity extraction method: ``"rule_based"`` uses regex heuristics.
    device : str
        Compute device for models.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        generator_model: str = "gpt-3.5-turbo",
        chunk_size: int = 512,
        entity_extract_method: str = "rule_based",
        device: str = "cpu",
    ) -> None:
        self._embedding_model_name = embedding_model
        self._generator_model_name = generator_model
        self._chunk_size = chunk_size
        self._entity_extract_method = entity_extract_method
        self._device = device

        self._graph: dict[str, dict[str, list[str]]] = {}
        self._entity_chunks: dict[str, list[int]] = {}
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

        self._embedder: Any = None
        self._generator: Any = None

        logger.info("GraphRAGPipeline initialized (lazy model loading)")

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from flashrag.embeddings.sentence_transformer import (
                SentenceTransformerEmbedding,
            )

            self._embedder = SentenceTransformerEmbedding(
                self._embedding_model_name, device=self._device
            )
        return self._embedder

    def _get_generator(self) -> Any:
        if self._generator is None:
            from flashrag.generation.generator import RAGGenerator

            self._generator = RAGGenerator(
                model_name=self._generator_model_name, device=self._device
            )
        return self._generator

    # ------------------------------------------------------------------
    # Entity and relation extraction
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[tuple[str, str]]:
        """
        Extract (entity, entity_type) tuples from text using rule-based patterns.

        Detects capitalized multi-word phrases, quoted terms, and common
        proper-noun patterns.

        Parameters
        ----------
        text : str
            Source text to extract entities from.

        Returns
        -------
        list of tuple[str, str]
            Extracted entities with their inferred type.
        """
        entities: list[tuple[str, str]] = []
        seen: set[str] = set()

        cap_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        for match in cap_pattern.finditer(text):
            name = match.group(1).strip()
            if name not in seen and len(name) > 2:
                entities.append((name, "PROPER_NOUN"))
                seen.add(name)

        quoted_pattern = re.compile(r'"([^"]{2,50})"')
        for match in quoted_pattern.finditer(text):
            term = match.group(1).strip()
            if term not in seen:
                entities.append((term, "QUOTED_TERM"))
                seen.add(term)

        acronym_pattern = re.compile(r"\b([A-Z]{2,6})\b")
        for match in acronym_pattern.finditer(text):
            acr = match.group(1)
            if acr not in seen and acr not in {"THE", "AND", "FOR", "NOT"}:
                entities.append((acr, "ACRONYM"))
                seen.add(acr)

        single_cap = re.compile(r"\b([A-Z][a-z]{2,})\b")
        for match in single_cap.finditer(text):
            word = match.group(1)
            if word not in seen and not text.startswith(word):
                entities.append((word, "NOUN"))
                seen.add(word)

        return entities

    def _extract_relations(
        self, text: str, entities: list[tuple[str, str]]
    ) -> list[tuple[str, str, str]]:
        """
        Extract (subject, predicate, object) triples from text.

        Uses proximity-based heuristics: when two entities co-occur in the
        same sentence, attempts to identify the connecting verb phrase.

        Parameters
        ----------
        text : str
            Source text.
        entities : list of tuple[str, str]
            Previously extracted entities.

        Returns
        -------
        list of tuple[str, str, str]
            Relation triples (subject, predicate, object).
        """
        if len(entities) < 2:
            return []

        sentence_re = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_re.split(text)
        entity_names = [e[0] for e in entities]
        triples: list[tuple[str, str, str]] = []

        verb_pattern = re.compile(
            r"\b(is|are|was|were|has|have|had|uses|used|contains|provides|"
            r"includes|creates|produces|requires|supports|enables|involves|"
            r"represents|describes|defines|implements|extends|builds|"
            r"connects|links|relates to|depends on|works with)\b",
            re.IGNORECASE,
        )

        for sent in sentences:
            present = [e for e in entity_names if e in sent]
            if len(present) < 2:
                continue

            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    subj, obj = present[i], present[j]
                    subj_pos = sent.find(subj)
                    obj_pos = sent.find(obj)

                    if subj_pos > obj_pos:
                        subj, obj = obj, subj
                        subj_pos, obj_pos = obj_pos, subj_pos

                    between = sent[subj_pos + len(subj): obj_pos].strip()
                    verb_match = verb_pattern.search(between)

                    if verb_match:
                        predicate = verb_match.group(0).lower()
                    else:
                        predicate = "related_to"

                    triples.append((subj, predicate, obj))

        return triples

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, chunks: list[Chunk]) -> None:
        """
        Build knowledge graph from entity/relation extraction over chunks.

        Parameters
        ----------
        chunks : list of Chunk
            Indexed document chunks.
        """
        self._graph = {}
        self._entity_chunks = {}

        for idx, chunk in enumerate(chunks):
            entities = self._extract_entities(chunk.text)
            triples = self._extract_relations(chunk.text, entities)

            for entity, _ in entities:
                entity_key = entity.lower()
                if entity_key not in self._entity_chunks:
                    self._entity_chunks[entity_key] = []
                self._entity_chunks[entity_key].append(idx)

            for subj, pred, obj in triples:
                subj_key = subj.lower()
                obj_key = obj.lower()

                if subj_key not in self._graph:
                    self._graph[subj_key] = {}
                if pred not in self._graph[subj_key]:
                    self._graph[subj_key][pred] = []
                if obj_key not in self._graph[subj_key][pred]:
                    self._graph[subj_key][pred].append(obj_key)

                if obj_key not in self._graph:
                    self._graph[obj_key] = {}
                inv_pred = f"inv_{pred}"
                if inv_pred not in self._graph[obj_key]:
                    self._graph[obj_key][inv_pred] = []
                if subj_key not in self._graph[obj_key][inv_pred]:
                    self._graph[obj_key][inv_pred].append(subj_key)

        logger.info(
            f"Knowledge graph built: {len(self._graph)} entities, "
            f"{sum(len(r) for rels in self._graph.values() for r in rels.values())} "
            f"relation edges"
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(
        self,
        paths: list[str | Path] | None = None,
        texts: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        Load, chunk, extract entities/relations, and build graph + vector index.

        Parameters
        ----------
        paths : list of str or Path, optional
            File paths to load and index.
        texts : list of str, optional
            Raw text documents to index.
        metadata : list of dict, optional
            Metadata for each document.

        Returns
        -------
        int
            Number of indexed chunks.
        """
        from flashrag.data.preprocessor import Preprocessor

        preprocessor = Preprocessor(chunk_size=self._chunk_size)

        if paths:
            self._chunks = preprocessor.process_files(paths)
        elif texts:
            self._chunks = preprocessor.process_texts(texts, metadata)
        else:
            raise ValueError("Provide either 'paths' or 'texts'")

        chunk_texts = [c.text for c in self._chunks]
        embedder = self._get_embedder()
        logger.info(f"Encoding {len(chunk_texts)} chunks for graph_rag...")
        self._vectors = np.asarray(embedder.encode(chunk_texts, show_progress=True))

        self._build_graph(self._chunks)

        logger.info(f"GraphRAG indexed {len(self._chunks)} chunks")
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Graph search
    # ------------------------------------------------------------------

    def _graph_search(self, query: str, top_k: int = 5, depth: int = 2) -> list[str]:
        """
        Retrieve relevant text via entity matching and graph traversal.

        Identifies entities in the query, finds them in the graph, performs
        BFS up to *depth* hops, collects source chunk indices for all
        reachable entities, and returns their texts.

        Parameters
        ----------
        query : str
            The search query.
        top_k : int
            Maximum number of passages to return.
        depth : int
            Maximum BFS traversal depth.

        Returns
        -------
        list of str
            Retrieved passage texts from graph traversal.
        """
        query_entities = self._extract_entities(query)
        query_tokens = set(query.lower().split())

        seed_entities: list[str] = []
        for entity, _ in query_entities:
            key = entity.lower()
            if key in self._graph or key in self._entity_chunks:
                seed_entities.append(key)

        if not seed_entities:
            for token in query_tokens:
                if len(token) > 3:
                    for entity_key in self._graph:
                        if token in entity_key or entity_key in token:
                            seed_entities.append(entity_key)
                            break
                if len(seed_entities) >= 3:
                    break

        if not seed_entities:
            return []

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        for e in seed_entities:
            queue.append((e, 0))
            visited.add(e)

        reachable_entities: list[str] = list(seed_entities)

        while queue:
            entity, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            if entity in self._graph:
                for _rel, targets in self._graph[entity].items():
                    for target in targets:
                        if target not in visited:
                            visited.add(target)
                            reachable_entities.append(target)
                            queue.append((target, current_depth + 1))

        chunk_indices: set[int] = set()
        for entity in reachable_entities:
            if entity in self._entity_chunks:
                chunk_indices.update(self._entity_chunks[entity])

        scored: list[tuple[int, float]] = []
        for idx in chunk_indices:
            chunk_tokens = set(self._chunks[idx].text.lower().split())
            overlap = len(query_tokens & chunk_tokens)
            scored.append((idx, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scored[:top_k]]

        return [self._chunks[idx].text for idx in top_indices]

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return (chunk_index, score) pairs from cosine similarity search."""
        if self._vectors is None or len(self._vectors) == 0:
            return []

        embedder = self._get_embedder()
        query_vec = np.asarray(embedder.encode([query]))[0]

        norms = np.linalg.norm(self._vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        similarities = self._vectors @ query_vec / (norms * query_norm + 1e-10)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        top_k: int = 5,
        graph_depth: int = 2,
        **gen_kwargs: Any,
    ) -> RAGResult:
        """
        Execute the GraphRAG pipeline: graph search + vector retrieval → generate.

        Parameters
        ----------
        question : str
            The user question.
        top_k : int
            Number of passages to retrieve.
        graph_depth : int
            Maximum BFS depth for graph traversal.

        Returns
        -------
        RAGResult
            Generated answer with retrieved contexts and metadata.
        """
        graph_passages = self._graph_search(question, top_k=top_k, depth=graph_depth)

        vector_results = self._vector_search(question, top_k=top_k)
        vector_passages = [self._chunks[idx].text for idx, _ in vector_results]
        vector_scores = [score for _, score in vector_results]

        seen_texts: set[str] = set()
        combined_passages: list[str] = []
        combined_scores: list[float] = []

        for passage in graph_passages:
            normalized = passage.strip()[:200]
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                combined_passages.append(passage)
                combined_scores.append(1.0)

        for passage, score in zip(vector_passages, vector_scores):
            normalized = passage.strip()[:200]
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                combined_passages.append(passage)
                combined_scores.append(score)

        combined_passages = combined_passages[:top_k]
        combined_scores = combined_scores[:top_k]

        generator = self._get_generator()
        context_block = "\n\n---\n\n".join(combined_passages)

        gen_result = generator.generate(
            question=question,
            context=context_block,
            **gen_kwargs,
        )

        return RAGResult(
            answer=gen_result.answer if hasattr(gen_result, "answer") else str(gen_result),
            contexts=combined_passages,
            sources=[{"graph_retrieved": i < len(graph_passages)}
                     for i in range(len(combined_passages))],
            scores=combined_scores,
            metadata={
                "graph_entities_found": len(
                    self._extract_entities(question)
                ),
                "graph_passages": len(graph_passages),
                "vector_passages": len(vector_passages),
            },
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_entity_summary(self, entity: str) -> str:
        """
        Get a text summary of an entity's relations in the knowledge graph.

        Parameters
        ----------
        entity : str
            The entity name to look up.

        Returns
        -------
        str
            Human-readable summary of the entity's connections.
        """
        key = entity.lower()
        if key not in self._graph:
            return f"Entity '{entity}' not found in the knowledge graph."

        lines: list[str] = [f"Entity: {entity}"]
        relations = self._graph[key]
        for rel, targets in relations.items():
            if rel.startswith("inv_"):
                display_rel = f"(inverse) {rel[4:]}"
            else:
                display_rel = rel
            lines.append(f"  —[{display_rel}]→ {', '.join(targets)}")

        if key in self._entity_chunks:
            lines.append(
                f"  Appears in {len(self._entity_chunks[key])} chunk(s)"
            )

        return "\n".join(lines)

    def get_graph_stats(self) -> dict[str, Any]:
        """
        Return summary statistics about the knowledge graph.

        Returns
        -------
        dict
            Dictionary with entity count, edge count, and chunk count.
        """
        total_edges = sum(
            len(targets)
            for rels in self._graph.values()
            for targets in rels.values()
        )
        unique_relations: set[str] = set()
        for rels in self._graph.values():
            for rel in rels:
                if not rel.startswith("inv_"):
                    unique_relations.add(rel)

        return {
            "num_entities": len(self._graph),
            "num_edges": total_edges,
            "num_relation_types": len(unique_relations),
            "num_chunks": len(self._chunks),
            "entities_with_chunks": len(self._entity_chunks),
        }
