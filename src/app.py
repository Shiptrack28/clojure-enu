"""
Gradio UI for the Research RAG Assistant.
Run: python src/app.py
"""

import gradio as gr
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.pipeline import ResearchRAG, Document

rag = ResearchRAG(top_k=6)
_ingested = False

DEMO_DOCS = [
    {
        "title": "RAG vs Fine-tuning Survey",
        "source": "arxiv_rag_survey.pdf",
        "text": """Retrieval-augmented generation (RAG) has emerged as a leading approach for
        knowledge-intensive NLP tasks. Unlike fine-tuning which permanently modifies model weights,
        RAG retrieves relevant documents at inference time, enabling dynamic knowledge updates
        without retraining. Studies consistently show RAG reduces hallucination by 30-40% on
        factual QA benchmarks compared to closed-book models. The key tradeoff is latency:
        retrieval adds 200-800ms depending on index size and retriever type."""
    },
    {
        "title": "Dense vs Sparse Retrieval Analysis",
        "source": "retrieval_comparison_2024.pdf",
        "text": """Dense retrieval using bi-encoder transformers outperforms BM25 on semantic
        similarity tasks by 8-15% NDCG@10 on BEIR benchmark. However, BM25 retains a
        significant advantage on exact-match keyword queries (7% better recall@10).
        Hybrid retrieval combining both methods via Reciprocal Rank Fusion (RRF) achieves
        the best overall performance, with consistent improvements over either method alone
        across 18 evaluated datasets. Cross-encoder reranking further improves precision
        by 12% but increases latency by 250-400ms."""
    },
    {
        "title": "LLM Hallucination Mitigation Strategies",
        "source": "hallucination_survey.pdf",
        "text": """Hallucination in large language models manifests as confident generation of
        factually incorrect statements. Three main mitigation strategies have been identified:
        (1) RAG grounding with retrieved documents reduces hallucination by 34% on average,
        (2) Chain-of-thought prompting reduces reasoning errors by 22%, and (3) Constitutional
        AI and RLHF alignment reduces harmful outputs but has less impact on factual errors.
        Combining RAG with chain-of-thought shows the highest accuracy on knowledge-intensive tasks."""
    },
    {
        "title": "Production RAG System Design",
        "source": "production_rag_guide.pdf",
        "text": """Production RAG systems require careful attention to chunking strategy, embedding
        model selection, vector index configuration, and monitoring. Chunk size of 256-512 tokens
        with 10-15% overlap is recommended for most document types. OpenAI text-embedding-3-small
        provides the best quality-cost tradeoff for most applications. For vector databases,
        FAISS with HNSW index is recommended for <1M vectors, while Pinecone or Qdrant suit
        larger deployments. Monitoring should track faithfulness, relevancy, and latency p95
        on every production query."""
    },
]


def load_demo_docs():
    global _ingested
    if _ingested:
        return "Already loaded"
    for doc in DEMO_DOCS:
        rag.ingest_text(doc["text"], source=doc["source"], title=doc["title"])
    _ingested = True
    return f"Loaded {len(DEMO_DOCS)} research papers ({len(rag._index)} chunks indexed)"


def format_answer(result) -> Tuple[str, str]:
    """Format answer and citations for display."""
    conf_pct = f"{result.confidence:.0%}"
    conf_color = "#10b981" if result.confidence > 0.6 else "#f59e0b" if result.confidence > 0.4 else "#ef4444"

    # Answer section
    answer_html = f"""
<div style="background:#0d1b2a; border:1px solid #1e3a5f44; border-radius:10px; padding:1.2rem; margin-bottom:1rem;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
  <span style="color:#94a3b8; font-size:0.85rem; font-family:monospace">
    {result.latency_ms:.0f}ms · {result.sources_consulted} sources · {len(result.sub_queries_used)} queries
  </span>
  <span style="color:{conf_color}; font-weight:700; font-size:0.9rem">
    Confidence: {conf_pct}
  </span>
</div>
<div style="color:#e2e8f0; line-height:1.7; font-size:0.95rem">
{result.answer}
</div>
</div>"""

    if result.contradictions:
        for c in result.contradictions:
            answer_html += f'<div style="background:#1a0a0a;border-left:3px solid #f59e0b;padding:0.5rem 0.8rem;margin:0.3rem 0;border-radius:4px;color:#fcd34d;font-size:0.8rem">⚠️ {c}</div>'

    # Citations
    citations_html = ""
    for cit in result.citations:
        score_color = "#10b981" if cit["score"] > 0.5 else "#f59e0b" if cit["score"] > 0.3 else "#64748b"
        citations_html += f"""
<div style="background:#0a1220; border:1px solid #1e3a5f33; border-radius:8px; padding:0.8rem 1rem; margin:0.4rem 0;">
  <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
    <span style="color:#3b82f6; font-weight:600; font-size:0.85rem">[Source {cit['source_num']}] {cit['title']}</span>
    <span style="color:{score_color}; font-size:0.8rem; font-family:monospace">score: {cit['score']:.3f}</span>
  </div>
  <div style="color:#64748b; font-size:0.78rem; font-family:monospace; margin-bottom:0.4rem">{cit['source']}</div>
  <div style="color:#94a3b8; font-size:0.82rem; line-height:1.5; font-style:italic">"{cit['excerpt']}"</div>
</div>"""

    return answer_html, citations_html


def ask(question: str, use_multi_query: bool = True):
    if not _ingested:
        load_demo_docs()
    if not question.strip():
        return "<p style='color:#ef4444'>Please enter a question.</p>", ""

    try:
        result = rag.answer(question.strip(), use_multi_query=use_multi_query)
        answer_html, citations_html = format_answer(result)
        sub_q_text = "\n".join(f"• {q}" for q in result.sub_queries_used)
        return answer_html, citations_html, sub_q_text
    except Exception as e:
        return f"<p style='color:#ef4444'>Error: {str(e)}</p>", "", ""


EXAMPLES = [
    "What are the key advantages of RAG over fine-tuning for knowledge-intensive tasks?",
    "Compare dense retrieval and BM25 sparse retrieval",
    "How does reranking improve RAG precision and what is the latency tradeoff?",
    "What strategies reduce LLM hallucination in production systems?",
    "What chunk size is recommended for production RAG deployments?",
]

CSS = """
body { background: #050a14; }
.gradio-container { max-width: 1100px !important; }
"""

with gr.Blocks(title="Research RAG Assistant", css=CSS,
               theme=gr.themes.Base(primary_hue="blue")) as demo:
    gr.Markdown("""
# 📚 Research RAG Assistant
*Multi-document Q&A with citations, confidence scoring, and contradiction detection*
""")

    with gr.Row():
        with gr.Column(scale=5):
            question_input = gr.Textbox(
                label="Research Question",
                placeholder="Ask anything about the loaded research papers...",
                lines=2,
            )
        with gr.Column(scale=1):
            multi_query = gr.Checkbox(label="Multi-query expansion", value=True)
            ask_btn = gr.Button("🔍 Research", variant="primary")

    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[question_input])

    with gr.Row():
        with gr.Column(scale=3):
            answer_output = gr.HTML(label="Answer")
        with gr.Column(scale=2):
            gr.Markdown("**Sub-queries generated:**")
            sub_q_output = gr.Textbox(label="", lines=4, interactive=False)
            citations_output = gr.HTML(label="Citations")

    status = gr.Textbox(label="Status", interactive=False)

    demo.load(fn=load_demo_docs, outputs=[status])
    ask_btn.click(
        fn=ask,
        inputs=[question_input, multi_query],
        outputs=[answer_output, citations_output, sub_q_output],
    )
    question_input.submit(
        fn=ask,
        inputs=[question_input, multi_query],
        outputs=[answer_output, citations_output, sub_q_output],
    )

if __name__ == "__main__":
    demo.launch(server_port=7861, share=False)
