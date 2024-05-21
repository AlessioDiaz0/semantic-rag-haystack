import torch
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.utils import ComponentDevice
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from utils import read_input_json, serialize_generated_answer

# if torch is compiled with cuda support, we can offload the computation to the GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Arguably the most important part of the code. This is the "system prompt"
# LLAMA3 required a specific format, so do not change the special tags
chat_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Answer the query from the context provided.
If it is possible to answer the question from the context, copy the answer from the context.
If the answer in the context isn't a complete sentence, make it one.

Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %};
<eot_id><start_header_id|>user<|end_header_id|>
query: {{query}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer:
"""

# defining the generator using quantized llama.cpp model
# lots of these parameters can have a significant impact
# it is best to read up on the parameters before changing them
generator = LlamaCppGenerator(
    model="models/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
    n_batch=512,
    model_kwargs={"n_gpu_layers": -1, "n_predict": -1},
    generation_kwargs={
        "max_tokens": 200,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9,
    },
)

# # Uses the sentence-transformers library to embed the text of the query
# Different models should be tested to see which one works best for the use case
text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/distiluse-base-multilingual-cased-v1", device=ComponentDevice(device)
)

generator.warm_up()
text_embedder.warm_up()

# initialize the document store and provide the path of the ingested
# documents
document_store = ChromaDocumentStore(persist_path="chromaDB")


rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
# Uses the ChromaEmbeddingRetriever to retrieve the top k documents
# from the document store based on the embeddings of the query
# top_k can be adjusted to see what works best
rag_pipeline.add_component(
    "embedder_retriever",
    ChromaEmbeddingRetriever(document_store=document_store, top_k=5),
)
# Uses the TransformersSimilarityRanker to rank the retrieved documents
# based on the similarity of the query with the documents
# The model can be changed to see what works best
rag_pipeline.add_component(
    "ranker",
    TransformersSimilarityRanker(
        model="cross-encoder/ms-marco-MiniLM-L-12-v2", device=ComponentDevice(device)
    ),
)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=chat_template))
rag_pipeline.add_component("llm", generator)
rag_pipeline.add_component("answer_builder", AnswerBuilder())

# It is easier to visualize the connections by viewing the respective
# pipeline image in the visual design folder
rag_pipeline.connect("text_embedder", "embedder_retriever")
rag_pipeline.connect("embedder_retriever", "ranker")
rag_pipeline.connect("ranker", "prompt_builder")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("llm.meta", "answer_builder.meta")
rag_pipeline.connect("embedder_retriever", "answer_builder.documents")


rag_pipeline.draw("visual_design/rag_pipeline.png")

prompts = read_input_json("documents/input/input.json")
results = []
for prompt_dict in prompts:
    question = prompt_dict["question"]
    answer = prompt_dict["answer"]
    # run the pipeline with the question as the input
    result = rag_pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"query": question},
            "answer_builder": {"query": question},
            "ranker": {"query": question},
        }
    )
    result_with_original = {"answer": answer, "generated_answer": result}
    results.append(result_with_original)

serialize_generated_answer(results)

if __name__ == '__main__':

    run
