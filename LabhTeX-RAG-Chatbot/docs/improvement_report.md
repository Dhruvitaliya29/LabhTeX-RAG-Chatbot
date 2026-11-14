# RAG Chatbot Improvement Report

## 1. Objective

Evaluate the effect of hybrid retrieval (BM25 + FAISS) compared with the original FAISS semantic retrieval pipeline.

The same 10 evaluation questions were used for both methods, covering direct, combined, and unsupported questions.

## 2. Experimental Configuration

| Component | Configuration |
|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector database | FAISS IndexFlatIP |
| Keyword retrieval | BM25 (hybrid only) |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) |
| LLM provider | Groq |
| Retrieval top K | 5 |
| Similarity threshold | 0.25 |

## 3. Evaluation Methodology

- Ten identical questions were evaluated using both retrieval methods.
- Questions were categorized as direct, combined, and unsupported.
- Both result files reference the same three evaluation document filenames.
- Answers were manually reviewed for correctness.
- Latency was recorded in seconds for each question.
- Latency includes retrieval and LLM generation and may vary with API and network conditions.

## 4. Quantitative Results

| Metric | FAISS baseline | Hybrid retrieval |
|---|---:|---:|
| Questions evaluated | 10 | 10 |
| Manually reviewed correctness | 10/10 | 10/10 |
| Total retrieved chunks | 25 | 45 |
| Average chunks per question | 2.5 | 4.5 |
| Average latency | 0.739 s | 4.979 s |
| Unsupported Q9 refusals | 1/1 | 1/1 |
| Unsupported Q10 refusals | 1/1 | 1/1 |

## 5. Analysis of Results

### Answer correctness

Both methods achieved 10/10 manually reviewed correct answers in this clean-corpus evaluation. Hybrid retrieval did not improve answer correctness over the FAISS baseline.

### Retrieval volume

Hybrid retrieval returned 45 chunks compared with 25 for the baseline, an increase of 80%. It returned an average of 4.5 chunks per question compared with 2.5 for FAISS.

Retrieving more chunks does not automatically mean retrieving more useful information. Additional context can increase processing overhead without improving the final answer.

### Latency

The hybrid method recorded an average latency of 4.979 seconds, compared with 0.739 seconds for FAISS. This is approximately 6.74 times the baseline latency in these runs.

The hybrid evaluation included Reciprocal Rank Fusion and keyword retrieval, which add processing steps. API and network variability may also affect end-to-end latency.

### Unsupported questions

Both methods refused the two unsupported questions in the manually reviewed evaluation. However, Q10 still retrieved chunks, demonstrating that retrieval results alone do not establish whether sufficient evidence exists to answer a question.

## 6. Implemented Improvement

Hybrid retrieval was implemented by combining BM25 keyword retrieval with FAISS semantic retrieval using Reciprocal Rank Fusion (RRF).

This provides a second retrieval signal that can help identify keyword-specific matches alongside semantic similarity.

In this experiment, hybrid retrieval maintained the same manually reviewed answer correctness but increased retrieval volume and latency. Therefore, it demonstrates an implemented alternative retrieval strategy, not a measured accuracy improvement.

## 7. Proposed Future Improvements

### Improvement 1: Retrieval reranking

Apply a cross-encoder reranker to the candidate chunks returned by semantic or hybrid retrieval. This could prioritize context that is more relevant to the complete user question and reduce irrelevant context passed to the LLM.

### Improvement 2: Adaptive retrieval and caching

Use a smaller candidate set or adaptive top-k selection when sufficient evidence is available. Cache repeated query embeddings and frequently requested retrieval results to reduce repeated computation.

## 8. Limitations

- Evaluation used only 10 questions, so the results are not sufficient to establish general performance across different document collections.
- Answer correctness was assessed manually rather than using an independent automated evaluation framework.
- Matching source filenames does not independently establish that the underlying document contents and index configurations were identical during both runs.
- End-to-end latency is affected by external LLM API and network variability.
- Retrieved chunk count is not a direct measure of retrieval relevance or answer quality.
- Citation display was tested, but citation precision and completeness have not yet been independently scored.

## 9. Conclusion

Both retrieval methods achieved 10/10 manually reviewed correct answers in the clean-corpus experiment. Hybrid retrieval increased the number of retrieved chunks by 80% and recorded approximately 6.74 times the baseline latency.

The experiment did not demonstrate an accuracy improvement from hybrid retrieval. FAISS semantic retrieval was more efficient in these particular runs. Further testing with a larger question set, retrieval relevance measurements, and reranking is needed before selecting a universally better retrieval strategy.

