import os

import torch
from haystack import Pipeline
from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# if torch is compiled with cuda support, we can offload the computation to the GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

folder_path = "documents/source_documents"

source_documents_folder = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

# Initialize the document store and provide a persistant path to store the documents
document_store = ChromaDocumentStore(persist_path="chromaDB")

ingest_pipeline = Pipeline()

# Uses the sentence-transformers library to embed the text of the documents
# Different models should be tested to see which one works best for the use case
ingest_pipeline.add_component(
    "document_embedder",
    SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/distiluse-base-multilingual-cased-v1", device=ComponentDevice(device)
    ),
)
ingest_pipeline.add_component(
    "document_writer",
    DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE),
)
ingest_pipeline.add_component(
    "file_type_router",
    FileTypeRouter(mime_types=["application/pdf", "text/markdown", "text/plain"]),
)
# The document splitter component is used to split the documents into smaller parts
# By default the text is split every 200 words
# We should tune the split_by and split_length parameters to see what works best
ingest_pipeline.add_component(
    "document_splitter",
    DocumentSplitter(),
)
# Document cleaner has some good default values and we can expand
# on it with regex patterns to clean the text if needed
ingest_pipeline.add_component("document_cleaner", DocumentCleaner())
ingest_pipeline.add_component("document_joiner", DocumentJoiner())
ingest_pipeline.add_component("pdf_converter", PyPDFToDocument())
ingest_pipeline.add_component("markdown_converter", MarkdownToDocument())
ingest_pipeline.add_component("text_file_converter", TextFileToDocument())

# It is easier to visualize the connections by viewing the respective
# pipeline image in the visual design folder
ingest_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
ingest_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
ingest_pipeline.connect("file_type_router.application/pdf", "pdf_converter.sources")
ingest_pipeline.connect("text_file_converter", "document_joiner")
ingest_pipeline.connect("markdown_converter", "document_joiner")
ingest_pipeline.connect("pdf_converter", "document_joiner")
ingest_pipeline.connect("document_joiner", "document_cleaner")
ingest_pipeline.connect("document_cleaner", "document_splitter")
ingest_pipeline.connect("document_splitter", "document_embedder")
ingest_pipeline.connect("document_embedder", "document_writer")

ingest_pipeline.draw("visual_design/ingest_pipeline.png")
ingest_pipeline.run({"file_type_router": {"sources": source_documents_folder}})
print(document_store.count_documents())
