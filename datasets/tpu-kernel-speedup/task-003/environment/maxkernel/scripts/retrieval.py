#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone CLI for querying a Vertex AI RAG Engine corpus."""

import argparse
import os
import sys

import vertexai
from vertexai import rag

DEFAULT_PROJECT = os.environ.get("RAG_PROJECT", "tpu-kernel-assist-sandbox")
DEFAULT_LOCATION = os.environ.get("RAG_LOCATION", "us-west1")
DEFAULT_CORPUS = os.environ.get(
    "RAG_CORPUS",
    "projects/tpu-kernel-assist-sandbox/locations/us-west1/ragCorpora/7991637538768945152",
)
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "10"))
DEFAULT_DISTANCE_THRESHOLD = float(
    os.environ.get("RAG_DISTANCE_THRESHOLD", "0.6"))


def retrieve(
    query: str,
    project: str = DEFAULT_PROJECT,
    location: str = DEFAULT_LOCATION,
    corpus: str = DEFAULT_CORPUS,
    top_k: int = DEFAULT_TOP_K,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
):
    """Runs a single retrieval query against the RAG corpus and returns the response."""
    vertexai.init(project=project, location=location)
    return rag.retrieval_query(
        text=query,
        rag_resources=[rag.RagResource(rag_corpus=corpus)],
        rag_retrieval_config=rag.RagRetrievalConfig(
            top_k=top_k,
            filter=rag.Filter(vector_distance_threshold=distance_threshold),
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Query a Vertex AI RAG corpus.")
    parser.add_argument("query",
                        help="Text query to retrieve relevant context for.")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--distance-threshold",
                        type=float,
                        default=DEFAULT_DISTANCE_THRESHOLD)
    args = parser.parse_args()

    response = retrieve(
        args.query,
        project=args.project,
        location=args.location,
        corpus=args.corpus,
        top_k=args.top_k,
        distance_threshold=args.distance_threshold,
    )

    contexts = response.contexts.contexts
    if not contexts:
        print(f"No matching result found for query: {args.query!r}")
        return

    results = [context.text for context in contexts]
    print(f"Vertex AI RAG Search Results for '{args.query}':\n\n" +
          "\n---\n".join(results))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error querying RAG corpus: {e}", file=sys.stderr)
        sys.exit(1)
