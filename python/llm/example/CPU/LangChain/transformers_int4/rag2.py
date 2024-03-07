#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.

# Code is adapted from https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_multi_modal_RAG_LLaMA2.ipynb
import time
import uuid
import argparse
from contextlib import contextmanager
from typing import Any, List

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from bigdl.llm.langchain.llms import TransformersLLM

# Constants and paths
PATH = "./Papers/Callifusion.pdf"
OUTPUT_FILE_PATH = "timing_results.txt"
MODEL_ID = '/mnt/disk1/models/vicuna-7b-v1.5'

# Helper class
class Element(BaseModel):
    type: str
    text: Any

# Context manager for timing
@contextmanager
def timeit(task_name, output_file_path):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    with open(output_file_path, "a") as file:
        file.write(f"Task {task_name} - Execution time: {elapsed_time} seconds\n")

# Function to partition PDF and get elements
def get_pdf_elements(path):
    return partition_pdf(
        filename=path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )

# Function to categorize elements from PDF
def categorize_elements(raw_elements):
    categorized_elements = []
    for element in raw_elements:
        if "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))
    return categorized_elements

# Function to generate summaries using LLM
def generate_summaries(text_elements, model):
    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text.\
    Table or text chunk: {element}"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    texts = [i.text for i in text_elements if i.text != ""]
    return summarize_chain.batch(texts, {"max_concurrency": 5})

# Function to check and print empty summaries
def check_and_print_empty_summaries(summaries, summary_type):
    empty_indices = [i for i, summary in enumerate(summaries) if not summary]
    if empty_indices:
        print(f"Empty {summary_type} summaries at indices:", empty_indices)
    else:
        print(f"No empty {summary_type} summaries found")

# Function to setup retriever with summaries
def setup_retriever(texts, text_summaries):
    vectorstore = Chroma(
        collection_name="summaries", embedding_function=GPT4AllEmbeddings()
    )

    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries) if s
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))
    return retriever

# Main execution
def main(model_path, file_path, question):
    # Partition and categorize PDF elements
    raw_elements = get_pdf_elements(file_path)
    categorized_elements = categorize_elements(raw_elements)
    text_elements = [e for e in categorized_elements if e.type == "text"]

    # Initialize and configure the model
    model = TransformersLLM.from_model_id(
        model_id=model_path,
        model_kwargs={"temperature": 0, "max_length": 1500, "trust_remote_code": True},
    )

    # Generate text summaries
    output_file_path = "timing_results.txt"
    with timeit("text_summaries", output_file_path):
        text_summaries = generate_summaries(text_elements, model)

    # Check for empty summaries
    check_and_print_empty_summaries(text_summaries, "text")

    # Set up the retriever with summaries
    retriever = setup_retriever([e.text for e in text_elements], text_summaries)

    # RAG pipeline
    template = """Answer the question based only on the following context, which can include text and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    with timeit("RAG pipeline", output_file_path):
        chain.invoke(question)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RAG pipeline on a given PDF with a specified question.')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('-p', '--file_path', type=str, required=True, help='Path to the PDF file')
    parser.add_argument('-q', '--question', type=str, required=False, help='Question to be asked to the model', 
                        default="What is the method of this paper?")
    args = parser.parse_args()

    main(args.model_path, args.file_path, args.question)