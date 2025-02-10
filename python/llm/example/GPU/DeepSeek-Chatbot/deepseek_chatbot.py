import os
import torch
import gradio as gr
from threading import Thread

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx2txt
except ImportError:
    docx2txt = None

############################################################################
# 1) Global config
############################################################################
MODEL_OPTIONS = [
    "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
]
current_model_name = MODEL_OPTIONS[0]

model_in_4bit = None
model_in_4bit_gpu = None
tokenizer = None

use_xpu = False
use_hybrid = False  # for CPU + XPU usage

DOC_TEXT = ""
MAX_TURNS_BEFORE_SUMMARY = 6


############################################################################
# 2) Document Parsing
############################################################################
def parse_uploaded_file(file_obj) -> str:
    global DOC_TEXT
    if file_obj is None:
        DOC_TEXT = ""
        return "No file uploaded."

    file_path = file_obj.name
    file_ext = os.path.splitext(file_path)[1].lower()
    extracted_text = ""

    if file_ext == ".pdf" and PyPDF2 is not None:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(len(reader.pages)):
                    page_text = reader.pages[page_num].extract_text() or ""
                    extracted_text += page_text
        except Exception as e:
            extracted_text = f"Error reading PDF: {e}"
    elif file_ext in [".doc", ".docx"] and docx2txt is not None:
        try:
            extracted_text = docx2txt.process(file_path) or ""
        except Exception as e:
            extracted_text = f"Error reading Word: {e}"
    else:
        extracted_text = f"Unsupported file extension or missing libraries for {file_ext}"

    DOC_TEXT = extracted_text.strip()
    if not DOC_TEXT or "Error reading" in extracted_text:
        return f"Document text is empty or invalid.\nExtracted: {extracted_text[:300]}"
    else:
        snippet = DOC_TEXT[:300].replace("\n", " ")
        return (
            f"Successfully parsed document. Text length = {len(DOC_TEXT)}.\n"
            f"Preview:\n{snippet}..."
        )

############################################################################
# 3) Retrieval logic, Summarization, etc.
############################################################################


def naive_retrieve_snippet(query: str, doc_text: str, window=300) -> str:
    query_lower = query.lower()
    text_lower = doc_text.lower()
    idx = text_lower.find(query_lower)
    if idx == -1:
        return ""
    else:
        start = max(0, idx - window // 2)
        end = min(len(doc_text), idx + window // 2)
        return doc_text[start:end]


def summarize_history(history, model, tok):
    """
    Summarize the entire conversation so far into a concise 'memory.'
    We'll call the same model for a quick summary.
    """
    # Convert history into a text block
    conversation_text = []
    for user_msg, bot_msg in history:
        conversation_text.append(f"User: {user_msg}")
        conversation_text.append(f"Assistant: {bot_msg if bot_msg else ''}")
    convo_str = "\n".join(conversation_text)

    # Build a simple prompt for summarizing
    summary_prompt = (
        "System: Please summarize the following conversation in 2-3 sentences. "
        "Focus on key details, topics, or questions.\n\n"
        f"Conversation:\n{convo_str}\n\nSummary:"
    )

    # We'll do a blocking generate call with minimal tokens. No streaming needed.
    input_ids = tok([summary_prompt], return_tensors='pt')
    if use_xpu or use_hybrid:
        input_ids = input_ids.to('xpu')

    # We'll do a short generation (max_new_tokens ~ 64)
    output = model.generate(
        **input_ids,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.5
    )

    summary_text = tok.decode(
        output[0][input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    # Clean up the summary if needed
    return summary_text.strip()

############################################################################
# 4) System prompt & conversation format
############################################################################


def format_prompt(user_query, retrieved_context, chat_history):
    system_text = (
        "System: You DO have access to the user's document text. It is appended below. "
        "Use it to answer questions about the document. Please do not say you have no file access. "
        "If the user wants a summary, you can provide one. Always answer in English.\n\n"

        "## Few-Shot Example 1 (Multi-turn style)\n"
        "User: Hello, how are you?\n"
        "Assistant: Hello! I'm doing well, thank you. How can I assist you today?\n\n"
        "User: Can you remember what we discussed about GPUs?\n"
        "Assistant: Yes, you asked about the definition of a GPU previously, and I explained that it's "
        "a specialized processor designed for parallel computations.\n\n"

        "## Few-Shot Example 2 (Referring to doc snippet)\n"
        "User: I uploaded a document. Could you summarize it?\n"
        "Assistant: Certainly! Let me look at the document text appended. "
        "According to the snippet, it discusses advanced techniques in machine learning...\n\n"
        "User: Thanks. And can you highlight the part about regularization?\n"
        "Assistant: Sure! The doc states that dropout and weight decay are two forms of regularization...\n\n"

        "## End of Few-Shot Examples\n\n"
    )

    if retrieved_context.strip():
        system_text += f"Document snippet or full text:\n{retrieved_context}\n\n"
    else:
        system_text += "No relevant snippet found or doc is empty.\n\n"

    system_text += "Now continue the actual conversation:\n\n"

    prompt = [system_text]

    for user_msg, bot_msg in chat_history:
        if bot_msg is not None:
            prompt.append(f"User: {user_msg}\nAssistant: {bot_msg}\n\n")
        else:
            prompt.append(f"User: {user_msg}\nAssistant:")

    prompt.append(f"User: {user_query}\nAssistant:")
    return "".join(prompt)

############################################################################
# 5) Model loading & CPU+XPU logic
############################################################################


def load_model(model_name, cpu_embedding=False):
    """
    If cpu_embedding=True, we pass that param so that large embeddings
    can live on CPU, with the rest on XPU.
    """
    print(
        f"[*] Loading model: {model_name} with cpu_embedding={cpu_embedding}")
    m = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        load_in_4bit=True,
        cpu_embedding=cpu_embedding  # custom flag from ipex_llm.transformers if supported
    )
    t = AutoTokenizer.from_pretrained(model_name)
    return m, t


def set_model(model_name):
    global model_in_4bit, model_in_4bit_gpu, tokenizer, use_xpu, use_hybrid, current_model_name
    current_model_name = model_name

    # decide if we want cpu_embedding
    if use_hybrid:
        model_in_4bit, tokenizer_ = load_model(model_name, cpu_embedding=True)
    else:
        model_in_4bit, tokenizer_ = load_model(model_name, cpu_embedding=False)
    tokenizer = tokenizer_

    if use_xpu or use_hybrid:
        # Move the rest to XPU
        model_in_4bit_gpu = model_in_4bit.to("xpu")
    else:
        # CPU only
        model_in_4bit_gpu = model_in_4bit.to("cpu")

    return f"✅ Model loaded: {model_name}, hybrid={use_hybrid}"


def set_device(device_choice):
    """
    device_choice can be 'CPU', 'XPU', or 'CPU+XPU'
    """
    global use_xpu, use_hybrid, model_in_4bit_gpu, model_in_4bit

    if device_choice == "CPU":
        use_xpu = False
        use_hybrid = False
        model_in_4bit_gpu = model_in_4bit.to("cpu")
        return "✅ Using CPU only."
    elif device_choice == "XPU":
        use_xpu = True
        use_hybrid = False
        model_in_4bit_gpu = model_in_4bit.to("xpu")
        return "✅ Using XPU (Intel GPU)."
    else:  # "CPU+XPU"
        use_xpu = True
        use_hybrid = True
        # We'll reload the model with cpu_embedding=True
        # to place large embeddings on CPU. Then .to("xpu") for the rest.
        print(
            "[INFO] Re-loading model for hybrid usage: CPU embeddings + XPU for main layers.")
        # re-set the model
        set_model(current_model_name)
        return "✅ Using Hybrid: CPU embeddings + XPU."

############################################################################
# 6) Generation logic (streaming)
############################################################################


def generate_stream(input_ids, streamer, model):
    def run_generation():
        model.generate(
            **input_ids,
            streamer=streamer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1
        )
    thread = Thread(target=run_generation)
    thread.start()
    for new_text in streamer:
        yield new_text

############################################################################
# 7) Chat & Summaries
############################################################################


def chat(user_input, history):
    user_lower = user_input.lower()
    if ("summarize" in user_lower or "document" in user_lower) and DOC_TEXT:
        retrieved_context = DOC_TEXT
    else:
        retrieved_context = naive_retrieve_snippet(user_input, DOC_TEXT)

    prompt = format_prompt(user_input, retrieved_context, history)

    # Move input_ids to xpu if hybrid or xpu
    if use_xpu or use_hybrid:
        input_ids = tokenizer([prompt], return_tensors='pt').to("xpu")
    else:
        input_ids = tokenizer([prompt], return_tensors='pt')

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True)
    stream_output = generate_stream(input_ids, streamer, model_in_4bit_gpu)

    partial_bot_response = ""
    for piece in stream_output:
        partial_bot_response += piece
        updated_history = [list(item) for item in history]
        updated_history[-1][1] = partial_bot_response
        yield updated_history


def user(query, chat_history):
    new_history = chat_history + [[query, None]]
    if len(new_history) > MAX_TURNS_BEFORE_SUMMARY:
        summary = summarize_history(new_history, model_in_4bit_gpu, tokenizer)
        new_history = [("Memory summary of previous conversation", summary)]
    return "", new_history


############################################################################
# 8) Build the Gradio UI
############################################################################
print(f"[*] Loading default model: {MODEL_OPTIONS[0]}")
# default is CPU only
model_in_4bit, tokenizer = load_model(MODEL_OPTIONS[0], cpu_embedding=False)
model_in_4bit_gpu = model_in_4bit.to("cpu")

with gr.Blocks() as demo:
    gr.Markdown("## Hybrid CPU+XPU Option Example")

    with gr.Tabs():
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot()
            user_input = gr.Textbox(
                show_label=False, placeholder="Ask me anything...")
            suggestion_btn1 = gr.Button("What is a GPU?")
            suggestion_btn2 = gr.Button("Differences between CPU & GPU?")
            suggestion_btn3 = gr.Button("Summarize the uploaded document")

            suggestion_btn1.click(
                fn=lambda: "What is a GPU?", inputs=None, outputs=user_input)
            suggestion_btn2.click(
                fn=lambda: "What are the differences between CPU and GPU?", inputs=None, outputs=user_input)
            suggestion_btn3.click(
                fn=lambda: "Summarize the uploaded document", inputs=None, outputs=user_input)

            clear_btn = gr.Button("Clear Chat")

            user_input.submit(fn=user, inputs=[user_input, chatbot], outputs=[user_input, chatbot], queue=False).then(
                fn=chat,
                inputs=[user_input, chatbot],
                outputs=chatbot
            )

            def clear_fn():
                return []

            clear_btn.click(fn=clear_fn, inputs=[],
                            outputs=chatbot, queue=False)

        with gr.Tab("Upload Document"):
            file_input = gr.File(
                label="Upload PDF/Word (.pdf, .doc, .docx)", file_types=[".pdf", ".doc", ".docx"])
            upload_status = gr.Textbox(
                label="Upload Status", interactive=False)

            def handle_upload(file):
                return parse_uploaded_file(file)

            file_input.change(fn=handle_upload,
                              inputs=file_input, outputs=upload_status)

        with gr.Tab("Settings"):
            model_choice = gr.Dropdown(
                choices=MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Select Model", interactive=True)
            device_choice = gr.Radio(choices=[
                                     "CPU", "XPU", "CPU+XPU"], value="CPU", label="Select Device", interactive=True)
            model_status = gr.Textbox(
                value=f"Current Model: {MODEL_OPTIONS[0]}", label="Model Status", interactive=False)
            device_status = gr.Textbox(
                value="Current Device: CPU", label="Device Status", interactive=False)

            def update_model_box(selected_model):
                msg = set_model(selected_model)
                return f"Current Model: {selected_model} | {msg}"

            model_choice.change(fn=update_model_box,
                                inputs=model_choice, outputs=model_status)

            def update_device_box(device):
                msg = set_device(device)
                return f"Device: {device} | {msg}"

            device_choice.change(fn=update_device_box,
                                 inputs=device_choice, outputs=device_status)

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)
