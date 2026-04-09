"""
evaluation/shared/dataset_utils.py

Helpers to create or upsert LangSmith datasets and examples.

Uses langsmith.Client() which reads LANGSMITH_API_KEY and LANGSMITH_PROJECT
from the environment automatically (set via .env).
"""

import hashlib
import json

from langsmith import Client


def get_or_create_dataset(client: Client, name: str, description: str):
    """
    Return existing dataset by name, or create it if it does not exist.
    LangSmith dataset names are unique within a project.
    """
    existing = list(client.list_datasets(dataset_name=name))
    if existing:
        return existing[0]
    return client.create_dataset(dataset_name=name, description=description)


def upsert_examples(client: Client, dataset_id: str, examples: list[dict]) -> None:
    """
    Create examples in the dataset, skipping any whose metadata.example_id
    already exists (idempotent upsert).

    Each item in examples must be:
        {
            "inputs":   {...},
            "outputs":  {...},
            "metadata": {"example_id": "<stable-unique-string>"},
        }

    The example_id is used to detect duplicates. If not supplied, one is
    derived from a hash of the inputs dict.
    """
    existing_ids: set[str] = set()
    for ex in client.list_examples(dataset_id=dataset_id):
        meta = ex.metadata or {}
        eid = meta.get("example_id")
        if eid:
            existing_ids.add(eid)

    to_create = []
    for example in examples:
        meta = example.get("metadata", {})
        eid = meta.get("example_id") or _hash_inputs(example["inputs"])
        if eid in existing_ids:
            continue
        to_create.append({
            "inputs": example["inputs"],
            "outputs": example.get("outputs", {}),
            "metadata": {**meta, "example_id": eid},
        })

    if not to_create:
        print(f"  All {len(examples)} examples already exist in dataset — skipping.")
        return

    client.create_examples(
        inputs=[e["inputs"] for e in to_create],
        outputs=[e["outputs"] for e in to_create],
        metadata=[e["metadata"] for e in to_create],
        dataset_id=dataset_id,
    )
    print(f"  Created {len(to_create)} new example(s) in dataset.")


def _hash_inputs(inputs: dict) -> str:
    serialised = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]
