from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

# import nltk
# nltk.download('punkt')


# Path to save images
path = "./Papers/"

# Get elements
raw_pdf_elements = partition_pdf(
    filename=path + "Callifusion.pdf",
    # Using pdf format to find embedded image blocks
    extract_images_in_pdf=True,
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    # Hard max on chunks
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)

# Create a dictionary to store counts of each type
category_counts = {}

for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

# Unique_categories will have unique elements
# TableChunk if Table > max chars set above
unique_categories = set(category_counts.keys())
category_counts

class Element(BaseModel):
    type: str
    text: Any


# Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

# Text
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from bigdl.llm.langchain.llms import TransformersLLM

import time
from contextlib import contextmanager

@contextmanager
def timeit(task_name, output_file):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    with open(output_file, "a") as file:
        file.write(f"Task {task_name} - Execution time: {elapsed_time} seconds\n")

output_file_path = "timing_results.txt"

# Prompt
prompt_text = """You are an assistant tasked with summarizing tables and text. \
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

model = TransformersLLM.from_model_id(
    model_id='/mnt/disk1/models/vicuna-7b-v1.5',
    model_kwargs={"temperature": 0, "max_length": 1500, "trust_remote_code": True},
)

summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

with timeit("text_summaries", output_file_path):
    # Apply to text
    texts = [i.text for i in text_elements if i.text != ""]
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})

# Check for empty summaries
def check_and_print_empty_summaries(summaries, summary_type):
    empty_indices = [i for i, summary in enumerate(summaries) if not summary]
    if empty_indices:
        print(f"Empty {summary_type} summaries at indices:", empty_indices)
    else:
        print(f"No empty {summary_type} summaries found")

check_and_print_empty_summaries(text_summaries, "text")



import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="summaries", embedding_function=GPT4AllEmbeddings()
)

# The storage layer for the parent documents
store = InMemoryStore()  # <- Can we extend this to images
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries) if s
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))


from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Option 1: LLM
model = TransformersLLM.from_model_id(
    model_id='/mnt/disk1/models/vicuna-7b-v1.5',
    model_kwargs={"temperature": 0, "max_length": 2600, "trust_remote_code": True}
)

# Option 2: Multi-modal LLM
# model = LLaVA

def debug_print(output):
    print("Debug: Raw output:", output)
    with open("debug_print.txt", "w") as file:
        file.write(output)
    return output


# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

with timeit("RAG pipeline", output_file_path):
    print("Results:" + 
        chain.invoke(
        "What is the method of this paper?"
        )
    )