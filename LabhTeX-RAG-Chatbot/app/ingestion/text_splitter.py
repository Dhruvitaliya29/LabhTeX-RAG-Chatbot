import re


def split_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Invalid chunk_overlap")

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    def split_long_sentence(sentence):
        words = sentence.split()
        parts = []
        current_part = ""

        for word in words:
            candidate = f"{current_part} {word}".strip()

            if len(candidate) <= chunk_size:
                current_part = candidate
            else:
                if current_part:
                    parts.append(current_part)
                current_part = word

        if current_part:
            parts.append(current_part)

        return parts

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(split_long_sentence(sentence))
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            overlap_words = []
            if chunk_overlap:
                for word in reversed(current.split()):
                    candidate_overlap = " ".join(
                        [word] + overlap_words
                    )

                    if len(candidate_overlap) > chunk_overlap:
                        break

                    overlap_words.insert(0, word)

            overlap_text = " ".join(overlap_words)
            current = f"{overlap_text} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


def create_chunks(
    documents: list[dict],
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> list[dict]:

    chunks = []

    for document in documents:
        text_chunks = split_text(
            document["text"],
            chunk_size,
            chunk_overlap
        )

        for index, chunk in enumerate(text_chunks):
            chunks.append({
                "text": chunk,
                "source": document["source"],
                "page": document.get("page"),
                "chunk_id": (
                    f"{document['source']}_"
                    f"{document.get('page')}_{index}"
                )
            })

    return chunks
