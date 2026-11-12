import os
import re
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from groq import Groq
except Exception:
    Groq = None


class RAGEngine:
    """
    Lightweight RAG pipeline:
    documents -> chunks -> embeddings -> cosine retrieval -> grounded answer.
    Sentence Transformers creates dense semantic embeddings when available.
    TF-IDF is a transparent fallback if the model cannot be loaded.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.chunks: List[Dict[str, Any]] = []
        self.embedder = None
        self.vectorizer = None
        self.matrix = None
        self.embedding_mode = "not_loaded"
        self.model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.groq_client = None
        self.llm_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        api_key = os.getenv("GROQ_API_KEY")
        if api_key and Groq:
            self.groq_client = Groq(api_key=api_key)

    def _chunk_text(self, text: str, source: str, chunk_size: int = 85, overlap: int = 20):
        # Word-based chunking keeps short operational SOPs intact while supporting larger files.
        words = re.findall(r"\S+", text)
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            content = " ".join(words[start:end]).strip()
            chunks.append({
                "source": source,
                "text": content,
                "chunk_id": f"{Path(source).stem}-{len(chunks)+1}",
                "start_word": start,
                "end_word": end,
            })
            if end == len(words):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def load_knowledge_base(self):
        self.chunks = []
        for path in sorted(self.data_dir.glob("*.md")):
            raw = path.read_text(encoding="utf-8").strip()
            # Keep document heading and body together as source evidence.
            self.chunks.extend(self._chunk_text(raw, path.name))
        if not self.chunks:
            self.matrix = None
            return

        texts = [item["text"] for item in self.chunks]
        if SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer(self.model_name)
                self.matrix = self.embedder.encode(
                    texts, convert_to_numpy=True, normalize_embeddings=True
                )
                self.embedding_mode = f"dense_sentence_transformer:{self.model_name}"
                return
            except Exception as exc:
                print(f"SentenceTransformer unavailable; using TF-IDF fallback: {exc}")

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)
        self.embedding_mode = "sparse_tfidf_fallback"

    def _retrieve(self, question: str, top_k: int):
        if not self.chunks or self.matrix is None:
            return []
        if self.embedder is not None:
            q_vec = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
            scores = np.dot(self.matrix, q_vec[0])
            scores = np.asarray(scores).reshape(-1)
        else:
            q_vec = self.vectorizer.transform([question])
            scores = cosine_similarity(q_vec, self.matrix)[0]
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.chunks[int(i)], "score": float(scores[int(i)])}
            for i in order
        ]

    def _call_llm(self, question: str, evidence: List[Dict[str, Any]]) -> Optional[str]:
        if not self.groq_client:
            return None
        context = "\n\n".join(
            f"[Source: {item['source']} | Chunk: {item['chunk_id']}]\n{item['text']}"
            for item in evidence
        )
        system_prompt = (
            "You are an operational document assistant. Answer ONLY using the supplied context. "
            "Do not use outside knowledge, invent thresholds, or add procedures not stated in context. "
            "If the context does not contain the answer, say the documents do not provide enough information. "
            "For safety-related questions, repeat only the source's stated checks and say to follow approved site procedures. "
            "Be concise and cite source filenames in square brackets."
        )
        try:
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0.1,
                max_tokens=350,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"LLM request failed; using extractive fallback: {exc}")
            return None

    @staticmethod
    def _token_overlap(answer: str, evidence_text: str) -> float:
        stop = {"the","and","for","with","from","that","this","are","you","should","what","when","into","your","can","may","not","only"}
        a = {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", answer) if len(w) > 2 and w.lower() not in stop}
        e = {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", evidence_text) if len(w) > 2 and w.lower() not in stop}
        if not a:
            return 0.0
        return len(a & e) / len(a)


    def answer(self, question: str, top_k: int = 3):
        started = time.time()

        # Step 1: Retrieve the top matching chunks
        candidates = self._retrieve(question, top_k)

        # Step 2: Set the minimum retrieval threshold
        threshold = float(
            os.getenv(
                "RETRIEVAL_THRESHOLD",
                "0.12" if "tfidf" in self.embedding_mode else "0.25"
            )
        )

        # Step 3: Find the highest similarity score
        best_score = candidates[0]["score"] if candidates else 0.0

        # Step 4: Keep only sufficiently relevant evidence
        evidence = [
            item for item in candidates
            if item["score"] >= threshold
        ]

        # Step 5: Abstain if no relevant evidence is found
        if not evidence:
            return {
                "question": question,
                "answer": "I couldn't find sufficient supporting information in the provided Greenko documents. Please consult the approved operational procedure or responsible site team.",
                "supported": False,
                "guardrail": "abstained_low_retrieval_score",
                "retrieval_score": round(best_score, 4),
                "citations": [],
                "retrieval_evidence": [],
                "embedding_mode": self.embedding_mode,
                "llm_used": False,
                "latency_ms": round((time.time() - started) * 1000, 2),
            }

        # Step 6: Generate an answer using the LLM if configured
        llm_answer = self._call_llm(question, evidence)

        if llm_answer:
            answer_text = llm_answer
            llm_used = True
        else:
            # Extractive fallback
            answer_text = (
                "Based on the retrieved operational document: "
                + " ".join(item["text"] for item in evidence[:2])
            )
            llm_used = False

        # Step 7: Check answer grounding
        evidence_text = " ".join(item["text"] for item in evidence)
        overlap = self._token_overlap(answer_text, evidence_text)

        grounded = overlap >= 0.20 and best_score >= threshold

        if not grounded:
            answer_text = (
                "The retrieved documents may be relevant, but I cannot verify "
                "that a complete answer is supported by their text. "
                "Please review the cited evidence and approved site procedure."
            )

        # Step 8: Prepare citations
        citations = []
        seen = set()

        for item in evidence:
            if item["source"] not in seen:
                citations.append({
                    "source": item["source"],
                    "chunk_id": item["chunk_id"]
                })
                seen.add(item["source"])

        # Step 9: Return the final response
        return {
            "question": question,
            "answer": answer_text,
            "supported": bool(grounded),
            "guardrail": (
                "passed_heuristic_grounding_check"
                if grounded
                else "flagged_possible_unsupported_content"
            ),
            "grounding_overlap": round(overlap, 4),
            "retrieval_score": round(best_score, 4),
            "citations": citations,
            "retrieval_evidence": [
                {
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "score": round(item["score"], 4),
                    "text": item["text"],
                }
                for item in evidence
            ],
            "embedding_mode": self.embedding_mode,
            "llm_used": llm_used,
            "latency_ms": round((time.time() - started) * 1000, 2),
        }


    def list_documents(self):
        return [
            {"filename": p.name, "characters": len(p.read_text(encoding="utf-8"))}
            for p in sorted(self.data_dir.glob("*.md"))
        ]

    def status(self):
        return {
            "document_count": len(self.list_documents()),
            "chunk_count": len(self.chunks),
            "embedding_mode": self.embedding_mode,
            "llm_configured": self.groq_client is not None,
        }

    def evaluate(self):
        csv_path = self.data_dir / "questions.csv"
        if not csv_path.exists():
            return {"message": "questions.csv not found", "tests": []}
        tests = []
        hits = 0
        total = 0
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            print("CSV HEADER:", next(csv.reader([csv_path.open(encoding="utf-8").readline()])))
            for row in csv.DictReader(f):
                question = row.get("question", "").strip()
                expected = row.get("expected_source", "").strip()
                if not question or not expected:
                    continue
                result = self.answer(question, top_k=3)
                returned = [c["source"] for c in result["citations"]]
                hit = expected in returned
                hits += int(hit)
                total += 1
                tests.append({
                    "question": question,
                    "expected_source": expected,
                    "retrieved_sources": returned,
                    "source_hit": hit,
                    "supported": result["supported"],
                    "retrieval_score": result["retrieval_score"],
                })
        return {
            "total_questions": total,
            "source_hit_rate": round(hits / total, 4) if total else 0,
            "note": "Source hit rate is measured against the expected_source labels in the evaluation CSV. This is a small prototype evaluation dataset.",

            "tests": tests,
        }
