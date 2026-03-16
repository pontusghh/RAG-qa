from pathlib import Path

import yaml

from rag.generate import generate_answer, generate_lm_only_answer
from rag.ingest import build_vector_store, load_pubmedqa
from rag.retrieve import retrieve_documents


CONFIG_PATH = "config/config.yaml"
DATASET_PATH_OVERRIDE = None
SAMPLE_SIZE = 200
RANDOM_SEED = 42
NUM_EXAMPLES = 3


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calc_accuracy(gold: list[str], preds: list[str]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for g, p in zip(gold, preds) if g == p)
    return correct / len(gold)


def calc_f1_yes(gold: list[str], preds: list[str]) -> float:
    tp = sum(1 for g, p in zip(gold, preds) if g == "yes" and p == "yes")
    fp = sum(1 for g, p in zip(gold, preds) if g == "no" and p == "yes")
    fn = sum(1 for g, p in zip(gold, preds) if g == "yes" and p == "no")

    precision_den = tp + fp
    recall_den = tp + fn
    if precision_den == 0 or recall_den == 0:
        return 0.0

    precision = tp / precision_den
    recall = tp / recall_den
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_rag(questions, documents, vector_store, config: dict, num_examples: int = 3):
    print("\n=== Evaluating RAG on the dataset ===")
    gold = []
    preds = []

    total_docs = 0
    hits_any = 0
    recall1 = 0
    recall3 = 0
    recall5 = 0

    example_count = 0

    for _, row in questions.iterrows():
        q = row["question"]
        gold_label = row["gold_label"]
        gold_doc_id = int(row["gold_document_id"])

        retrieved_docs, retrieved_ids = retrieve_documents(
            vector_store,
            q,
            top_k=config.get("retrieval", {}).get("top_k", 5),
        )
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        pred, raw_answer = generate_answer(q, context, config)

        if pred is None:
            continue

        gold.append(gold_label)
        preds.append(pred)

        if retrieved_ids:
            total_docs += 1
            if gold_doc_id in retrieved_ids:
                hits_any += 1
            if gold_doc_id in retrieved_ids[:1]:
                recall1 += 1
            if gold_doc_id in retrieved_ids[:3]:
                recall3 += 1
            if gold_doc_id in retrieved_ids[:5]:
                recall5 += 1

        if example_count < num_examples:
            print(f"\nRAG example: {example_count + 1}")
            print("Question:", q)
            print("Gold label:", gold_label)
            print("Model raw answer:", raw_answer)
            print("Extracted prediction:", pred)
            print("Gold document id:", gold_doc_id)

            gold_abstract = documents.loc[gold_doc_id].abstract
            print("\nGold document abstract:")
            print(gold_abstract[:500])

            for d in retrieved_docs:
                did = d.metadata.get("id")
                print(f"\nRetrieved doc id: {did}")
                print("Retrieved doc snippet:")
                print(d.page_content[:500], "...\n")
            example_count += 1

    acc = calc_accuracy(gold, preds)
    f1 = calc_f1_yes(gold, preds)

    print("\nRAG results:")
    print("  Valid predictions:", len(gold))
    print("  Accuracy:", round(acc, 4))
    print("  F1:", round(f1, 4))

    if total_docs > 0:
        print("\nRetrieval metrics:")
        print(f"  Gold retrieved anywhere: {hits_any} / {total_docs} ({hits_any/total_docs:.3f})")
        print(f"  Recall@1: {recall1/total_docs:.3f}")
        print(f"  Recall@3: {recall3/total_docs:.3f}")
        print(f"  Recall@5: {recall5/total_docs:.3f}")

    return {
        "valid_predictions": len(gold),
        "accuracy": acc,
        "f1_yes": f1,
        "retrieval_total": total_docs,
        "retrieval_hits_any": hits_any,
    }


def evaluate_lm_only(questions, config: dict, num_examples: int = 3):
    print("\n=== Baseline LM-only evaluation (no context) ===")

    gold = []
    preds = []

    for i, (_, row) in enumerate(questions.iterrows()):
        q = row["question"]
        gold_label = row["gold_label"]

        pred, raw_answer = generate_lm_only_answer(q, config)
        if pred is None:
            continue

        gold.append(gold_label)
        preds.append(pred)

        if i < num_examples:
            print(f"\nExample {i + 1}:")
            print("Question:", q)
            print("Gold label:", gold_label)
            print("Model raw answer:", raw_answer)
            print("Parsed prediction:", pred)

    acc = calc_accuracy(gold, preds)
    f1 = calc_f1_yes(gold, preds)

    print("\nLM-only results:")
    print("  Valid predictions:", len(gold))
    print("  Accuracy:", round(acc, 4))
    print("  F1:", round(f1, 4))

    return {
        "valid_predictions": len(gold),
        "accuracy": acc,
        "f1_yes": f1,
    }


def main():
    config = load_config(CONFIG_PATH)
    dataset_path = DATASET_PATH_OVERRIDE or config.get("paths", {}).get("default_dataset", "./ori_pqal.json")

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    documents, questions = load_pubmedqa(dataset_path)

    if SAMPLE_SIZE and SAMPLE_SIZE < len(questions):
        questions = questions.sample(SAMPLE_SIZE, random_state=RANDOM_SEED)

    vector_store = build_vector_store(documents, config)

    rag_metrics = evaluate_rag(
        questions=questions,
        documents=documents,
        vector_store=vector_store,
        config=config,
        num_examples=NUM_EXAMPLES,
    )

    lm_metrics = evaluate_lm_only(
        questions=questions,
        config=config,
        num_examples=NUM_EXAMPLES,
    )

    print("\n=== Summary ===")
    print("RAG:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rag_metrics.items()})
    print("LM-only:", {k: round(v, 4) if isinstance(v, float) else v for k, v in lm_metrics.items()})


if __name__ == "__main__":
    main()
