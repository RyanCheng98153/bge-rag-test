# pip install transformers torch llama-index llama-index-indices-managed-bge-m3
# pip install llama-index-postprocessor-flag-embedding-reranker
# Usage: python bge_rag.py


# Use a pipeline as a high-level helper
from transformers import AutoTokenizer, AutoModelForCausalLM
# rag dependencies
from llama_index.core import Settings
from llama_index.core import Document
from llama_index.indices.managed.bge_m3 import BGEM3Index
from llama_index.core.postprocessor import LLMRerank
# from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

def generate_text(question, knowledge):
    
    prompt = f""" You are a professional Question Answering System.
    Answer the question the given knowledge. 
    ----------------------------------------
    knowledge: {knowledge}
    ----------------------------------------
    Question: {question}
    """
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def get_retrieved_nodes(
    query_str, vector_top_k=10, reranker_top_n=3, with_reranker=False
):
    Settings.chunk_size = 8192
    # Let's create some demo corpus
    sentences = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]
    
    documents = [Document(doc_id=str(i), text=s) for i, s in enumerate(sentences)]
    # Indexing with BGE-M3 model
    index = BGEM3Index.from_documents(
        documents,
        weights_for_different_modes=[
            0.4,
            0.2,
            0.4,
        ],  # [dense_weight, sparse_weight, multi_vector_weight]
    )

    query_bundle = QueryBundle(query_str)
    
    # configure retriever
    # retriever = VectorIndexRetriever(
    #     index=index,
    #     similarity_top_k=vector_top_k,
    # )
    
    index = BGEM3Index.from_documents(
        documents,
        weights_for_different_modes=[
            0.4,
            0.2,
            0.4,
        ],  # [dense_weight, sparse_weight, multi_vector_weight]
    )
    retriever = index.as_retriever(similarity_top_k=vector_top_k)
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        # configure reranker
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

    return retrieved_nodes

if __name__ == "__main__":
    # Load model directly
    question = "What is BGE M3?"
    knowledge = ""
    
    response = generate_text(question=question, knowledge=knowledge)
    print("response:", response)
    
    # Define a retriever
    nodes = get_retrieved_nodes(
        query_str=question,
        vector_top_k=3,
        with_reranker=False,    
    )
    
    knowledge = nodes[0].text
    print("knowledge:", knowledge)
    
    response = generate_text(question=question, knowledge=knowledge)
    print("response:", response)
    
    nodes = get_retrieved_nodes(
        query_str=question,
        vector_top_k=3,
        with_reranker=True,    
    )
    
    knowledge = nodes[0].text
    print("knowledge:", knowledge)
    