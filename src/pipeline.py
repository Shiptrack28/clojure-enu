"""
Research RAG pipeline — multi-document Q&A with citations, confidence scoring,
and source attribution. Built for research/academic use cases (not crypto).

Features over basic RAG:
- Multi-query retrieval: expands one question to 3 sub-queries
- Confidence scoring: tells you when it's uncertain
- Citation formatting: every claim linked to a source
- Contradiction detection: flags when sources disagree
"""

import os
import time
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL   = "gpt-4o-mini"


@dataclass
class Document:
    text: str
    source: str
    title: str
    page: Optional[int] = None
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text[:100].encode()).hexdigest()[:8]


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    rank: int


@dataclass
class ResearchAnswer:
    question: str
    answer: str
    citations: List[Dict]          # [{source, title, excerpt, chunk_id}]
    confidence: float              # 0–1
    sub_queries_used: List[str]
    sources_consulted: int
    contradictions: List[str]      # warnings when sources disagree
    latency_ms: float


def embed_texts(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [item.embedding for item in response.data]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    import math
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    return dot / (mag_a * mag_b + 1e-9)


class ResearchRAG:
    """
    Multi-document research assistant with citations and confidence.

    Design decisions:
    - Multi-query expansion: one question → 3 sub-queries → more recall
    - Deduplication: remove near-duplicate chunks before sending to LLM
    - Confidence: measured by average retrieval score + source count
    - Citations: structured with source + excerpt for every claim
    """

    def __init__(self, top_k: int = 6, similarity_threshold: float = 0.25):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self._index: List[Tuple[Document, List[float]]] = []  # (doc, embedding)

    def ingest(self, documents: List[Document]):
        """Embed and index a list of documents."""
        print(f"Indexing {len(documents)} chunks...")
        texts = [d.text for d in documents]
        embeddings = embed_texts(texts)
        for doc, emb in zip(documents, embeddings):
            self._index.append((doc, emb))
        print(f"Indexed {len(self._index)} chunks total")

    def ingest_text(self, text: str, source: str, title: str,
                    chunk_size: int = 400, overlap: int = 50):
        """Chunk a raw text string and ingest it."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            if chunk_text:
                chunks.append(Document(
                    text=chunk_text,
                    source=source,
                    title=title,
                    page=i // chunk_size + 1,
                ))
        self.ingest(chunks)
        return len(chunks)

    def _retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Dense retrieval for a single query."""
        if not self._index:
            return []
        k = top_k or self.top_k
        query_emb = embed_texts([query])[0]
        scored = [
            (doc, cosine_similarity(query_emb, emb))
            for doc, emb in self._index
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(document=doc, score=score, rank=i)
            for i, (doc, score) in enumerate(scored[:k])
            if score >= self.similarity_threshold
        ]

    def _expand_query(self, question: str) -> List[str]:
        """Generate 2 sub-queries to improve recall via multi-query retrieval."""
        prompt = (
            f"Given this research question, generate 2 different sub-questions "
            f"that would help retrieve relevant information. Return ONLY the 2 questions, "
            f"one per line, no numbering.\n\nQuestion: {question}"
        )
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=120,
            )
            sub_qs = [q.strip() for q in resp.choices[0].message.content.strip().split("\n")
                      if q.strip()]
            return [question] + sub_qs[:2]
        except Exception:
            return [question]

    def _deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Remove near-duplicate chunks (same source, overlapping text)."""
        seen_texts = set()
        unique = []
        for chunk in chunks:
            # Use first 80 chars as dedup key
            key = chunk.document.text[:80].lower().strip()
            if key not in seen_texts:
                seen_texts.add(key)
                unique.append(chunk)
        return unique

    def _detect_contradictions(self, chunks: List[RetrievedChunk]) -> List[str]:
        """
        Simple contradiction detection — flag when different sources
        use strongly opposing language on the same topic.
        (Lightweight proxy; real version uses NLI model)
        """
        opposing_pairs = [
            ("increase", "decrease"), ("effective", "ineffective"),
            ("significant", "not significant"), ("safe", "dangerous"),
            ("positive", "negative"), ("confirms", "refutes"),
        ]
        contradictions = []
        texts = [c.document.text.lower() for c in chunks]
        sources = [c.document.source for c in chunks]

        for word_a, word_b in opposing_pairs:
            sources_with_a = [sources[i] for i, t in enumerate(texts) if word_a in t]
            sources_with_b = [sources[i] for i, t in enumerate(texts) if word_b in t]
            if sources_with_a and sources_with_b and set(sources_with_a) != set(sources_with_b):
                contradictions.append(
                    f"Sources may disagree: '{word_a}' ({sources_with_a[0]}) "
                    f"vs '{word_b}' ({sources_with_b[0]})"
                )
        return contradictions[:2]  # max 2 warnings

    def _compute_confidence(self, chunks: List[RetrievedChunk],
                             answer: str) -> float:
        """
        Confidence heuristic based on:
        - Average retrieval score (higher score = more relevant chunks)
        - Number of unique sources (more sources = higher confidence)
        - Answer length (very short = uncertain)
        """
        if not chunks:
            return 0.1

        avg_score = sum(c.score for c in chunks) / len(chunks)
        unique_sources = len(set(c.document.source for c in chunks))
        source_bonus = min(unique_sources * 0.08, 0.24)
        length_factor = min(len(answer.split()) / 100, 1.0)

        confidence = (avg_score * 0.6) + source_bonus + (length_factor * 0.16)
        return round(min(max(confidence, 0.0), 1.0), 3)

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(
                f"[Source {i+1}: {chunk.document.title} — {chunk.document.source}]\n"
                f"{chunk.document.text}"
            )
        return "\n\n---\n\n".join(parts)

    def answer(self, question: str, use_multi_query: bool = True) -> ResearchAnswer:
        """
        Full research pipeline:
        1. Expand question to sub-queries
        2. Retrieve for each sub-query, merge and deduplicate
        3. Detect contradictions
        4. Generate answer with citations
        5. Compute confidence
        """
        start = time.time()

        # Step 1: Query expansion
        sub_queries = self._expand_query(question) if use_multi_query else [question]

        # Step 2: Multi-query retrieval + deduplicate
        all_chunks: List[RetrievedChunk] = []
        for sq in sub_queries:
            all_chunks.extend(self._retrieve(sq, top_k=self.top_k))

        # Sort by score, deduplicate
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        top_chunks = self._deduplicate(all_chunks)[:self.top_k + 2]

        if not top_chunks:
            return ResearchAnswer(
                question=question, answer="No relevant documents found in the knowledge base.",
                citations=[], confidence=0.0, sub_queries_used=sub_queries,
                sources_consulted=0, contradictions=[],
                latency_ms=round((time.time()-start)*1000, 1),
            )

        # Step 3: Contradiction detection
        contradictions = self._detect_contradictions(top_chunks)

        # Step 4: Generate answer
        context = self._build_context(top_chunks)
        system = (
            "You are a precise research assistant. Answer the question using ONLY "
            "the provided source documents. For each factual claim, cite the source "
            "by referencing [Source N]. If the documents don't contain enough "
            "information, say so clearly. Do not hallucinate."
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        answer_text = resp.choices[0].message.content.strip()

        # Step 5: Build citations + confidence
        citations = [
            {
                "source_num": i + 1,
                "title": c.document.title,
                "source": c.document.source,
                "excerpt": c.document.text[:200] + "...",
                "score": round(c.score, 3),
                "chunk_id": c.document.chunk_id,
            }
            for i, c in enumerate(top_chunks)
        ]
        confidence = self._compute_confidence(top_chunks, answer_text)
        latency_ms = round((time.time() - start) * 1000, 1)

        return ResearchAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            sub_queries_used=sub_queries,
            sources_consulted=len(set(c.document.source for c in top_chunks)),
            contradictions=contradictions,
            latency_ms=latency_ms,
        )


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    rag = ResearchRAG(top_k=5)

    # Load sample docs if present, otherwise use built-in demo text
    demo_text = """
    Large language models have demonstrated remarkable capabilities in natural language
    understanding and generation. However, they are prone to hallucination — generating
    plausible-sounding but factually incorrect information. Retrieval-augmented generation
    (RAG) addresses this by grounding model responses in retrieved documents.

    Studies show that RAG significantly reduces hallucination rates compared to closed-book
    generation. In a 2023 benchmark, RAG systems achieved 34% lower hallucination rates
    than equivalent closed-book models on knowledge-intensive tasks.

    The quality of retrieval is critical. Dense retrieval using bi-encoder models generally
    outperforms sparse BM25 retrieval on semantic similarity tasks, but BM25 retains an
    advantage for exact keyword matching. Hybrid approaches combining both have shown the
    strongest overall performance across diverse query types.

    Reranking with cross-encoders further improves precision. Cross-encoders process query
    and document jointly, enabling more accurate relevance scoring at the cost of higher
    computational latency. The tradeoff between latency and quality must be evaluated
    based on application requirements.
    """

    n = rag.ingest_text(demo_text, source="demo_doc.txt", title="RAG Overview Paper")
    print(f"Ingested {n} chunks from demo document\n")

    question = sys.argv[1] if len(sys.argv) > 1 else "What are the benefits of RAG over closed-book generation?"
    result = rag.answer(question)

    print(f"Q: {result.question}")
    print(f"\nConfidence: {result.confidence:.1%}")
    print(f"Sub-queries used: {result.sub_queries_used}")
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nSources consulted: {result.sources_consulted}")
    print(f"Latency: {result.latency_ms}ms")
    if result.contradictions:
        print(f"\n⚠️  Contradictions detected: {result.contradictions}")
