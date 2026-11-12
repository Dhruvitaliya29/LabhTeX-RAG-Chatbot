# Baseline RAG Evaluation

## Configuration

- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Vector database: FAISS IndexFlatIP
- LLM provider: Groq
- Similarity threshold: 0.25
- Retrieval top K: 5
- Knowledge base: Sample RAG documents and uploaded resume PDFs

## Results

| ID | Category | Manual result |
|---|---|---|
| Q1 | Direct | Pass |
| Q2 | Direct | Pass |
| Q3 | Direct | Pass |
| Q4 | Direct | Pass |
| Q5 | Direct | Pass |
| Q6 | Combined | Pass |
| Q7 | Combined | Fail: unsupported implementation details |
| Q8 | Combined | Pass |
| Q9 | Unsupported | Pass: refused |
| Q10 | Unsupported | Pass: refused |

Preliminary manual answer correctness: 9/10.

Total retrieved chunks across all questions: 29.

Unsupported question refusal: 2/2.

## Key observations

1. Direct questions were generally answered correctly.
2. Unsupported questions were refused.
3. Q10 retrieved irrelevant chunks but the generator correctly refused.
4. Q7 contained unsupported claims about FastAPI, LangChain and the LLM implementation.
5. Resume documents sometimes appeared in retrieval results for unrelated RAG questions.

## Limitations

- This is a manually assessed preliminary evaluation.
- The knowledge base contains both sample RAG documents and personal resumes.
- Retrieval relevance and citation accuracy have not yet been independently scored.
- Latency measurements are environment-dependent.

## Planned improvements

1. Hybrid retrieval using BM25 and FAISS.
2. Stronger grounding instructions and citation validation.
