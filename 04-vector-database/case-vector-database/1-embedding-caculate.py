from model.embedding_model_langchain import embeddings

competition = embeddings.embeddings.create(
    model="text-embedding-v3",
    input="我想知道迪士尼的退票政策",
    dimensions=1024,
    encoding_format="float"
)

print(competition.model_dump_json())