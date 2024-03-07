# Langchain examples

The example here show how to use langchain RAG with `bigdl-llm`. This script rag2.py can analyze PDF file.

## Install bigdl-llm
Follow the instructions in [Install](https://github.com/intel-analytics/BigDL/tree/main/python/llm#install).

## Install Required Dependencies for langchain examples. 

```bash
pip install langchain unstructured[all-docs] lxml
pip install pydantic==1.10.9
pip install -U chromadb==0.3.25
pip install -U pandas==2.0.3
sudo apt-get update
sudo apt-get install tesseract-ocr
```

## Run the examples

### Semi Structured Multi Modal RAG

Before running the example, please download `punkt` by:
```python
import nltk
nltk.download('punkt')
```

```bash
python rag2.py -m MODEL_PATH -p PDF_PATH -q QUESTION
```
arguments info:
- `-m MODEL_PATH`: **required**, path to the model
- `-p PDF_PATH`: **required**, path to the PDF
- `-q QUESTION`: question to ask. Default is `What is the method of this paper?`.

#### Known Issues
When you have SSL issues related to huggingface.co, you can try using mirror site:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Example Output
```
Answer: The method of this paper is the development of a conditional diffusion model for generating Chinese calligraphy. The model uses a U-net model as the backbone and employs DDPMs sampling for the forward process (diffusion) and the reverse process (denoising). The style transfer technique is effective, and the system can generate artworks of any character, any script, and any style. The model can also perform style transfer via one-shot fine-tuning, which allows the transfer of scripts and styles to unseen characters and out-of-domain symbols. The fine-tuning technique is based on LoRA, which speeds up the training process of large models while reducing memitations of traditional learning methods and providing 
```


