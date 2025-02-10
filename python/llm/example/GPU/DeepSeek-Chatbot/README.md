# DeepSeek Chatbot

This is a hybrid CPU+XPU-enabled chatbot powered by DeepSeek models, optimized using the Intel IPEX-LLM library for efficient and fast inference on Intel AI PC systems. The chatbot supports large-model inference, document-based querying, and summarization capabilities. Users can easily toggle between different computation devices (CPU, XPU, or Hybrid) and select from various pre-trained DeepSeek models.

## Features

Model Selection: Supports multiple pre-trained DeepSeek models, including chat and base variants.
Device Flexibility: Offers CPU-only, GPU (XPU)-only, and hybrid CPU+XPU modes for optimal performance on Intel AI PCs.
Document Upload: Accepts PDF and Word documents to enable retrieval-augmented generation (RAG) for enhanced question answering.
Conversation Summarization: Summarizes long conversations to keep the chatbot's memory efficient and concise.
Interactive UI: Intuitive browser-based interface with separate tabs for chatting, document upload, and settings configuration.

## Installation

Requirements
Before running the chatbot, ensure you have the necessary dependencies installed:

Operating System: Compatible with Intel AI PCs running Linux or Windows.
Python Environment: Create and activate a new Conda environment with Python 3.11 and required libraries.
Steps to Install IPEX-LLM for Intel GPUs
Run the following commands in your terminal:

conda create -n llm python=3.11 libuv -y
conda activate llm
pip install --pre --upgrade ipex-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu

## Running the Chatbot

Steps to Launch:
Activate Environment:
conda activate llm
Navigate to Directory: Go to the directory containing the deepseek_chatbot.py file:
cd /path/to/DeepSeek-Chatbot
Start the Chatbot: Run the chatbot application:
python deepseek_chatbot.py
Access in Browser: Open your browser and navigate to:
http://127.0.0.1:7860/

## Features Overview

1. Chat Interface (Tab 1)
   Input questions or commands in the textbox.
   Suggested questions:
   "What is a GPU?"
   "What are the differences between CPU and GPU?"
   "Summarize the uploaded document."
2. Document Upload (Tab 2)
   Upload PDF or Word documents to allow the chatbot to extract and answer questions based on the document's content.
   Supported file formats: .pdf, .doc, .docx.
3. Settings (Tab 3)
   All models will be quantized to 4 bits.
   Select Model: Choose from pre-trained DeepSeek models, such as:
   deepseek-ai/deepseek-llm-7b-chat
   deepseek-ai/deepseek-llm-7b-base
   deepseek-ai/DeepSeek-R1-Distill-Qwen-14B (For this one you need to have enough RAM)

## Select Device:

User friendly to easily toggle between CPU and GPU.
CPU: Use the CPU for all computations.
XPU: Use the Intel GPU for inference.
CPU+XPU: Hybrid mode for large models, with embeddings on CPU and main layers on GPU.

## Known Limitations

Ensure the uploaded document's content is extractable (some scanned PDFs may not work).
Hybrid mode is optimized for large models but may require sufficient system memory.

## Additional Notes

For troubleshooting, ensure that the IPEX-LLM package is installed correctly and that your system supports Intel GPUs.
If the UI is not accessible at 127.0.0.1:7860, verify that the server is running and check for firewall restrictions.

## Future Enhancements:

This is an initial version demonstrating how easily IPEX-LLM enables running large language models on Intel AI PCs. Future updates will include the addition of more advanced models and enhanced Retrieval-Augmented Generation (RAG) capabilities (this current version is a naive retrieve snippet) to further improve the chatbot's performance and versatility. Stay tuned!
