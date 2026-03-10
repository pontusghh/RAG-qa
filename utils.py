from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState


class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.last_doc_ids: list[int] | None = None
        self.last_docs: list[Document] | None = None

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1] # get the user input query
        retrieved_docs = self.vector_store.similarity_search(
            last_message.text, k=5
        )  # search for documents

        # docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)  
        unique = {}
        for doc in retrieved_docs:
            doc_id = doc.metadata.get("id")
            if doc_id not in unique:
                unique[doc_id] = doc
        unique_docs = list(unique.values())
        docs_content = "\n\n".join(doc.page_content for doc in unique_docs)

        self.last_doc_ids = [doc.metadata.get("id") for doc in unique_docs]
        self.last_docs = unique_docs
        
        augmented_message_content = (
            "You are a biomedical expert. Given a question and context from a research paper, "
            "determine whether the answer is yes or no. Output ONLY 'yes' or 'no'.\n\n"
            "Example 1:\n"
            "Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\n"
            "Context: Programmed cell death (PCD) is the regulated death of cells within an organism. "
            "Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant, "
            "and highlight the correlation of this organelle with other organelles during developmental PCD.\n"
            "Answer: yes\n\n"
            "Example 2:\n"
            "Question: Does context matter for the relationship between deprivation and all-cause mortality?\n"
            "Context: The results suggest that the impact of socio-economic deprivation on mortality is not "
            "restricted to a few places.\n"
            "Answer: no\n\n"
            "Now answer the following:\n"
            f"Question: {last_message.text}\n"
            f"Context: {docs_content}\n\n"
            "Answer:"
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }