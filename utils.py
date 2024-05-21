import json
import os
import time
from dataclasses import asdict, is_dataclass

# from haystack_integrations.components.evaluators.uptrain import UpTrainMetric

# uptrain_parameters_mapping = {
#     UpTrainMetric.CONTEXT_RELEVANCE: ["questions", "contexts"],
#     UpTrainMetric.FACTUAL_ACCURACY: ["questions", "contexts", "responses"],
#     UpTrainMetric.RESPONSE_RELEVANCE: ["questions", "responses"],
#     UpTrainMetric.RESPONSE_COMPLETENESS: ["questions", "responses"],
#     UpTrainMetric.RESPONSE_COMPLETENESS_WRT_CONTEXT: [
#         "questions",
#         "contexts",
#         "responses",
#     ],
#     UpTrainMetric.RESPONSE_CONSISTENCY: ["questions", "contexts", "responses"],
#     UpTrainMetric.RESPONSE_CONCISENESS: ["questions", "responses"],
#     UpTrainMetric.CRITIQUE_LANGUAGE: ["responses"],
#     UpTrainMetric.CRITIQUE_TONE: ["responses"],
#     UpTrainMetric.GUIDELINE_ADHERENCE: [
#         "questions",
#         "responses",
#     ],
#     UpTrainMetric.RESPONSE_MATCHING: [
#         "questions",
#         "responses",
#         "ground_truths",
#     ],
# }


def serialize_generated_answer(results, parameters_info):
    serialized_data = []
    serialized_data.append(parameters_info)

    for result in results:
        answer = result["answer"]
        generated_answers = (
            result["generated_answer"].get("answer_builder", {}).get("answers", [])
        )

        for generated_answer in generated_answers:
            if is_dataclass(generated_answer):
                answer_dict = asdict(generated_answer)
                answer_dict["answer"] = answer

                if "documents" in answer_dict:
                    processed_documents = []
                    for doc in generated_answer.documents:
                        if is_dataclass(doc):
                            doc_dict = asdict(doc)
                            # Remove specified keys
                            doc_dict.pop("embedding", None)
                            doc_dict.pop("dataframe", None)
                            doc_dict.pop("blob", None)
                            doc_dict.pop("sparse_embedding", None)
                            processed_documents.append(doc_dict)
                    answer_dict["documents"] = processed_documents

                serialized_data.append(answer_dict)

    file_name = time.strftime("generated_data_%Y-%m-%d_%H-%M-%S.json")
    file_path = os.path.join("documents/generation_output", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, ensure_ascii=False, indent=2)


def read_serialized_generated_answer(file_name):
    queries = []
    documents = []
    answers = []
    staff_answers = []

    file_path = os.path.join("documents/generation_output", file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        serialized_data = json.load(f)
        for entry in serialized_data:
            query = entry.get("query", "")
            queries.append(query)

            document = [
                doc["content"] for doc in entry.get("documents", []) if "content" in doc
            ]
            documents.append(document)

            answer = entry.get("data", "")
            answers.append(answer)
            staff_answer = entry.get("answer", "")
            staff_answers.append(staff_answer)
    return queries, documents, answers, staff_answers


# def metric_to_params(metric, data):
#     params = {}
#     for param in uptrain_parameters_mapping.get(metric, []):
#         if param in data and data[param]:
#             params[param] = data[param]
#     return {"evaluator": params}


def serialize_evaluation_results(evaluation_results):
    file_name = time.strftime("evaluation_results_%Y-%m-%d_%H-%M-%S.json")
    file_path = os.path.join("documents/evaluation_output", file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)


def read_input_json(file_name):
    qa_pairs = []

    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            question = entry.get("questions", "")
            answer = entry.get("answer", "")
            qa_pairs.append({"question": question, "answer": answer})

    return qa_pairs
