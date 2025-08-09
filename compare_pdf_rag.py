#!/usr/bin/env python3
"""Compare RAG performance with different PDF extraction strategies.

This module offers two pathways to ingest a PDF and run Retrieval Augmented
Generation (RAG) over its contents:

1. **Metadata extraction** – text is taken directly from the PDF using
   :class:`PyPDF2.PdfReader`.
2. **Structured extraction with Mistral** – the PDF is partitioned with the
   `unstructured` library and each chunk is cleaned by a Mistral model.  This
   approach can handle scanned or poorly formatted PDFs.

For each ingestion method a vector store is built and used to answer a list of
questions.  The quality of the generated answers is assessed with the
`ragas` library which relies on an Azure OpenAI model to provide an impartial
evaluation.

The script is intentionally lightweight and focuses on clarity.  It is meant as
an educational example; real deployments should add error handling, caching and
more sophisticated prompt templates.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Dict


# The heavy imports are placed inside functions so that basic help/compilation
# works without the optional dependencies being installed.


# ---------------------------------------------------------------------------
# PDF ingestion strategies
# ---------------------------------------------------------------------------

def extract_metadata_text(pdf_path: Path) -> str:
    """Extract text directly from the PDF.

    This method relies on the PDF containing an embedded text layer.  For
    scanned documents where text is absent, the function will return an empty
    string.
    """
    from PyPDF2 import PdfReader  # local import to avoid hard dependency


    reader = PdfReader(str(pdf_path))
    pieces: List[str] = []
    for page in reader.pages:
        pieces.append(page.extract_text() or "")
    return "\n".join(pieces)

def extract_mistral_text(pdf_path: Path, api_key: str | None = None) -> str:

    """Use `unstructured` and a Mistral model to obtain cleaned text.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    api_key:
        API key for the `mistralai` service or compatible deployment. If not
        provided, the function looks up ``MISTRAL_API_KEY`` in the environment.

    """
    from unstructured.partition.pdf import partition_pdf
    from mistralai.client import MistralClient

    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY environment variable is required")


    elements = partition_pdf(filename=str(pdf_path))
    client = MistralClient(api_key=api_key)

    cleaned: List[str] = []
    for el in elements:
        prompt = (
            "Clean and structure the following text so it can be embedded for\n"
            "retrieval:\n" + el.text
        )
        resp = client.chat(
            model="mistral-small",
            messages=[{"role": "user", "content": prompt}],
        )
        cleaned.append(resp.output[0].content[0].text)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

def build_vector_store(text: str):
    """Create a FAISS vector store from raw text using Azure OpenAI embeddings."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import AzureOpenAIEmbeddings

    from langchain.vectorstores import FAISS

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

    embeddings = AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )

    return FAISS.from_documents(docs, embedding=embeddings)


def answer_questions(vector_store, questions: Iterable[str]) -> List[Dict[str, str]]:
    """Retrieve context and answer questions using a lightweight LLM.

    The LLM uses Azure OpenAI. Ensure the required environment variables are
    set, such as ``AZURE_OPENAI_API_KEY`` and ``AZURE_OPENAI_ENDPOINT``.
    Any `langchain` compatible LLM can be substituted.
    """
    from langchain.chains import RetrievalQA
    from langchain.llms import AzureOpenAI

    llm = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

    results: List[Dict[str, str]] = []
    for q in questions:
        out = qa(q)
        results.append({
            "question": q,
            "answer": out["result"],
            "contexts": [c.page_content for c in out["source_documents"]],
        })
    return results


# ---------------------------------------------------------------------------
# Evaluation with ragas
# ---------------------------------------------------------------------------

def evaluate_rag(rag_outputs: List[Dict[str, str]], truths: List[str]) -> Dict[str, float]:
    """Compute ragas metrics given model outputs and ground truth answers."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness
    from langchain.llms import AzureOpenAI

    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rag_outputs],
        "answer": [r["answer"] for r in rag_outputs],
        "contexts": [r["contexts"] for r in rag_outputs],
        "ground_truth": truths,
    })

    llm = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    )


    result = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness],
        llm=llm,
    )
    return {k: float(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument("questions", type=Path, help="Text file with one question per line")
    parser.add_argument("truths", type=Path, help="Text file with ground truth answers")
    parser.add_argument(
        "--method", choices=["metadata", "mistral"], default="metadata",
        help="PDF ingestion strategy",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = [q.strip() for q in args.questions.read_text().splitlines() if q.strip()]
    truths = [t.strip() for t in args.truths.read_text().splitlines() if t.strip()]

    if args.method == "metadata":
        text = extract_metadata_text(args.pdf)
    else:
        text = extract_mistral_text(args.pdf)


    store = build_vector_store(text)
    outputs = answer_questions(store, questions)
    scores = evaluate_rag(outputs, truths)

    for metric, value in scores.items():
        print(f"{metric}: {value:.3f}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
