import os
import re
import pandas as pd


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from sklearn.metrics import accuracy_score, f1_score

from utils import State, RetrieveDocumentsMiddleware

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def load_data():
    print("\n=== Loading data ===")
    tmp_data = pd.read_json("ori_pqal.json").T
    # some labels have been defined as "maybe", only keep the yes/no answers
    tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

    documents = pd.DataFrame({"abstract": tmp_data.apply(lambda row: (" ").join(row.CONTEXTS+[row.LONG_ANSWER]), axis=1),
                "year": tmp_data.YEAR})
    questions = pd.DataFrame({"question": tmp_data.QUESTION,
                "year": tmp_data.YEAR,
                "gold_label": tmp_data.final_decision,
                "gold_context": tmp_data.LONG_ANSWER,
                "gold_document_id": documents.index})
    print("Dataset length: ", len(questions))
    print("Example question:")
    print(questions.iloc[0].question)
    print("\nExample document abstract:")
    print(documents.iloc[0].abstract)

    return documents, questions

def load_model(model_id):
    print("\n=== Loading model ===")
    model = HuggingFacePipeline.from_model_id(
        model_id,
        task="text-generation",
        pipeline_kwargs={
            "return_full_text": False, 
            "max_new_tokens": 20,
            "temperature": 0.01,
            "do_sample": False,
        },
    )

    hf_pipe = getattr(model, "pipeline", None)
    device = None
    if hf_pipe is not None and hasattr(hf_pipe, "model"):
        device = getattr(hf_pipe.model, "device", None)

    print(f"Model id: {model_id}")
    print(f"Device: {device}")

    prompt = "What is the capital of France?"
    print("Example prompt:\n", prompt)
    answer = model.invoke(prompt)
    print("Answer:\n", answer)
    return model

# %%
def build_vector_store(documents):
    print("\n=== Building vector vector ===")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )

    metadatas = [{"id": idx} for idx in documents.index]
    texts = text_splitter.create_documents(
        texts=documents.abstract.tolist(), 
        metadatas=metadatas,
    )

    splits = text_splitter.split_documents(texts)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    print("Embedding length for 'Hello world':\n",len(embeddings.embed_query("Hello world")))

    # Initialize the Vector Store
    vector_store = Chroma(
        collection_name="pubmedqa_rag",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    # Add splits
    document_ids = vector_store.add_documents(documents=splits)

    print(f"Success! Added {len(splits)} chunks")
    print("\nTesting similarity search on 'What is programmed cell death?':")
    results = vector_store.similarity_search_with_score(
        "What is programmed cell death?", k=1
    )
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

    return vector_store


# %%
def build_agent(model, vector_store):
    print("\n=== Building agent ===")

    middleware = RetrieveDocumentsMiddleware(vector_store=vector_store)
    agent = create_agent(
        model,
        tools=[],
        middleware=[middleware],
    )

    print("Testing RAG agent:\n")
    for step in agent.stream(
        {"messages": [{"role": "user", "content": "Is programmed cell death the regulated death of cells?" }]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    
    return agent, middleware

# %%
def extract_yes_no(raw: str) -> str | None:
    text = raw.strip().lower()
    if not text:
        return None
    # First, try the first word
    first = text.split()[0]
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"
    # Fallback
    match = re.search(r'\b(yes|no)\b', text)
    if match:
        return match.group(1)
    return None

def rag_answer(agent, middleware, question: str):
    res = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    answer_msg = res["messages"][-1]
    answer_text = answer_msg.content
    pred = extract_yes_no(answer_text)
    
    retrieved_ids = middleware.last_doc_ids or []
    retrieved_docs = middleware.last_docs or []

    return pred, answer_text, retrieved_ids, retrieved_docs

# %%

def lm_only_answer(model, question: str):
    prompt = (
        "You are a biomedical expert. Answer the following yes/no question "
        "using your knowledge. Output ONLY 'yes' or 'no'.\n\n"
        "Example:\n"
        "Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\n"
        "Answer: yes\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    answer_text = model.invoke(prompt)
    pred = extract_yes_no(answer_text)
    return pred, answer_text

def evaluate_rag(agent, middleware, questions, documents, num_examples = 3):

    print("\n=== Evaluating RAG on the dataset ===")
    gold = []
    preds = []

    # retrieval statistics
    total_docs = 0
    hits_any = 0

    recall1 = 0
    recall3 = 0
    recall5 = 0

    example_count = 0

    for n, (i, row) in enumerate(questions.iterrows()):
        q = row["question"]
        gold_label = row["gold_label"]  # "yes"/"no"
        gold_doc_id = row["gold_document_id"]
        pred, raw_answer, retrieved_ids, retrieved_docs = rag_answer(agent, middleware, q)
        if pred is None:
            continue

        gold.append(gold_label)
        preds.append(pred)

        if retrieved_ids:
            total_docs += 1

            # check if gold doc appears anywhere
            if gold_doc_id in retrieved_ids:
                hits_any += 1

            # Recall@k calculations
            if gold_doc_id in retrieved_ids[:1]:
                recall1 += 1
            if gold_doc_id in retrieved_ids[:3]:
                recall3 += 1
            if gold_doc_id in retrieved_ids[:5]:
                recall5 += 1
        
        if example_count < num_examples:
            print("\nRAG example:", example_count + 1)
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
                print(f"\nRetrieved doc snippet:")
                print(d.page_content[:500], "...\n")
            example_count += 1
    
    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, pos_label="yes")
    print("\nRAG results:")
    print("  Valid predictions:", len(gold))
    print("  Accuracy:", acc)
    print("  F1:", f1)
    if total_docs > 0:
        print("\nRetrieval metrics:")
        print(f"  Gold retrieved anywhere: {hits_any} / {total_docs} ({hits_any/total_docs:.3f})")
        print(f"  Recall@1: {recall1/total_docs:.3f}")
        print(f"  Recall@3: {recall3/total_docs:.3f}")
        print(f"  Recall@5: {recall5/total_docs:.3f}")

def evaluate_lm_only(model, questions):

    gold = []
    preds = []

    print("\n=== Baseline LM-only evaluation (no context) ===")

    for i, (_, row) in enumerate(questions.iterrows()):
        q = row["question"]
        gold_label = row["gold_label"]

        pred, raw_answer = lm_only_answer(model, q)
        if pred is None:
            continue

        gold.append(gold_label)
        preds.append(pred)
        if i < 3:
            print(f"\nExample {i}:\n")
            print("\nQuestion:", q)
            print("Gold label:", gold_label)
            print("Model raw answer", raw_answer)
            print("Parsed prediction", pred)
        
    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, pos_label="yes")

    print("\nLM only (no context) results:")
    print("  Valid predictions:", len(gold))
    print("  Accuracy:", acc)
    print("  F1:", f1)

# %%

def main():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    documents, questions = load_data()
    questions = questions.sample(500, random_state=42)
    
    model = load_model(model_id)
    vector_store = build_vector_store(documents)
    agent, middleware = build_agent(model, vector_store)

    evaluate_rag(agent, middleware, questions, documents)
    evaluate_lm_only(model, questions)

# %%
if __name__ == "__main__":
    main()
