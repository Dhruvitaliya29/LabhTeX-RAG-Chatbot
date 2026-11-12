from groq import Groq
from app.config import GROQ_API_KEY, GROQ_MODEL


class GroqGenerator:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured.")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def generate_answer(self, question: str, retrieved_chunks: list[dict]):
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not retrieved_chunks:
            return {
                "answer": "I could not find sufficient information in the provided documents to answer this question.",
                "sources": [],
            }

        context_parts = []
        sources = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk["metadata"]

            source = {
                "citation": index,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "score": chunk.get("score"),
            }

            sources.append(source)

            context_parts.append(
                f"[Source {index}]\n"
                f"Filename: {source['source']}\n"
                f"Page: {source['page']}\n"
                f"Content: {chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        system_prompt = """You are a document-grounded question answering assistant.

Answer questions using only the supplied document context.

Rules:
1. Do not use outside knowledge or assumptions.
2. If the context does not contain enough evidence, clearly say that the documents do not provide sufficient information.
3. Do not invent facts, figures, names, or explanations.
4. Cite supporting information using [Source 1], [Source 2], etc.
5. Only use source numbers that actually appear in the supplied context.
6. Keep answers clear and directly relevant.
7. If only part of a question is supported, answer that part and identify what is missing."""

        user_prompt = (
            f"DOCUMENT CONTEXT:\n{context}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "Answer using the rules above."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_completion_tokens=1200,
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError("The LLM returned an empty response.")

        return {
            "answer": answer.strip(),
            "sources": sources,
        }
