from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

query_embedding = embeddings.embed_query("What is a sentence embedding?")
doc_embeddings = embeddings.embed_documents(
    [
        "Sentence embeddings map text to dense vectors.",
        "LangChain provides a standard Embeddings interface.",
    ]
)