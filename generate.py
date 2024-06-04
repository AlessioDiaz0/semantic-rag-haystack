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

import argparse

def generate(this_model, this_max_tokens, this_temperature,
             this_top_k, this_top_p, this_n_batch, this_prompt):

    # if torch is compiled with cuda support, we can offload the computation to the GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Arguably the most important part of the code. This is the "system prompt"
    # LLAMA3 required a specific format, so do not change the special tags

    chat_template = get_chat_template(this_prompt)

    # defining the generator using quantized llama.cpp model
    # lots of these parameters can have a significant impact
    # it is best to read up on the parameters before changing them
    generator = LlamaCppGenerator(
        model="models/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
        n_batch = this_n_batch,
        model_kwargs={"n_gpu_layers": -1, "n_predict": -1},
        generation_kwargs={
            "max_tokens": this_max_tokens,
            "temperature": this_temperature,
            "top_k": this_top_k,
            "top_p": this_top_p,
        },
    )

    # # Uses the sentence-transformers library to embed the text of the query
    # Different models should be tested to see which one works best for the use case
    text_embedder = SentenceTransformersTextEmbedder(
        model=f"sentence-transformers/{this_model}", device=ComponentDevice(device)
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
    parameters_info = (f"Parameters used: model {this_model}, max_tokens {this_max_tokens}, " +
                       f"temperature {this_temperature}, top_k {this_top_k}, top_p {this_top_p}, n_batch {this_n_batch}")
    
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

    serialize_generated_answer(results, parameters_info)


def collect_prompts():
    isPrompted = ''
    prompts = []
    while(isPrompted != 'y' and isPrompted != 'n'):
        isPrompted = (input("Would you like to provide QA's to the LLM with to improve accuracy? [Y/N]")).lower()

    if(isPrompted == 'y'):
        for i in range(3):
            display_prompt_samples()
            staff_prompt_q = input("Your question: ")
            staff_prompt_a = input("Your answer: ")
            prompts.append([staff_prompt_q, staff_prompt_a])
            if(i >= 2):
                more_prompts = (input(f"Would you like to continue ({i+1}/3)? [Y/N]")).lower()
            if(more_prompts == 'n'):
                break

    return prompts

def get_chat_template(prompts):
    
    if len(prompts) > 0:
        prompts_content = "\n        ".join([f"Question: {prompt[0]} Answer: {prompt[1]}" for prompt in prompts])

        chat_template = ("""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        """ +
        f"Here are some examples:\n        {prompts_content}"
        +
        """

        Context:
        {% for doc in documents %}
        {{ doc.content }}
        {% endfor %};
        <eot_id><start_header_id|>user<|end_header_id|>
        query: {{query}}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Answer:
        """)
    else:
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

    return chat_template


def display_prompt_samples():
    print("\n\nThese are some examples of good prompts:")
    print("\nQuestion: When is the midterm?")
    print("Answer: The midterm will be on June 6th")
    print("\nQuestion: What time are the professor office hours?")
    print("Answer: Office hours are held on Mondays and Thursdays 11:00AM to 12PM")
    print("\nQuestion: How much of the final counts towards our final grade?")
    print("Answer: The final is 30% of your grade\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'LLM parameters')

    parser.add_argument("-m", "--model", type = str, nargs= '+', default = ["distiluse-base-multilingual-cased-v1"],
                        help = "Name of the model Default: distiluse-base-multilingual-cased-v1")
    parser.add_argument("-maxt", "--max_tokens", type = int, nargs= '+', default = [200],
                        help = "Max tokens Default: 200")
    parser.add_argument("-temp", "--temperature", type = float, nargs= '+', default = [0.8],
                        help = "Tempeature values range from 0.0 to 2.0 Default: 0.8")
    parser.add_argument("-tk", "--top_k", type = int, nargs= '+', default = [40],
                        help = "Top_k value Default: 40")
    parser.add_argument("-tp", "--top_p", type = float, nargs= '+', default = [1.0],
                        help = "Top_p value Default: 1.0")
    parser.add_argument("-nb", "--n_batch", type = int, nargs= '+', default = [512],
                        help = "Number of batches Default: 512")
    parser.add_argument("-promp", "--add_prompt", type = str, default = [""],
                        help = "Allows stuff to input customed prompts")
    # parser.add_argument("-p", "--prompt", type = str, nargs= '+',
    #                     help = "Prompt given to the LLM Default: Read More")
    #Decided to not implement
    
    args = parser.parse_args()

    for model in args.model:
        for max_tokens in args.max_tokens:
            for temperature in args.temperature:
                for top_k in args.top_k:
                    for top_p in args.top_p:
                        for n_batch in args.n_batch:
                                prompts = collect_prompts()
                                generate(model, max_tokens, temperature, top_k, top_p, n_batch, prompts)

