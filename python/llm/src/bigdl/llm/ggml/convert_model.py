from bigdl.llm.ggml.convert import _convert_to_ggml
from bigdl.llm.ggml.quantize import quantize
from pathlib import Path
import time


def convert_model(input_path: str,
                  output_path: str = None,
                  model_family: str = 'llama',
                  dtype: str = 'int4'):
    """
    Convert Hugging Face llama-like / gpt-neox-like / bloom-like model to lower precision

    :param input_path: str, path of model, for example `./llama-7b-hf`.
    :param output_path: Save path of output quantized model. Default to `None`.
            If you don't specify this parameter, quantized model will be saved in
            the same directory as the input and just replace precision with quantize_type
            like `./ggml-model-q4_0.bin`.
    :param model_family: Which model family your input model belongs to. Default to `llama`.
            Now only `llama`/`bloomz`/`gptneox` are supported.
    :param dtype: Which quantized precision will be converted
            Now only int4 supported.
    """

    if dtype == 'int4':
        dtype = 'q4_0'

    model_name = Path(input_path).stem
    tmp_ggml_file_path = f'/tmp/{model_name}_{int(time.time())}'
    _convert_to_ggml(model_path=input_path,
                     outfile_dir=tmp_ggml_file_path,
                     model_family=model_family,
                     outtype="fp16")
    
    tmp_ggml_file_path = next(Path(tmp_ggml_file_path).iterdir())

    quantize(input_path=tmp_ggml_file_path,
             output_path=output_path,
             model_family=model_family,
             dtype=dtype)