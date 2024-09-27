# Whisper Test
The Whisper Test allows users to evaluate the performance and accuracy of [Whisper](https://huggingface.co/openai/whisper-base) speech-to-text models.
For accuracy, the model is tested on the [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) dataset using [Word Error Rate (WER)](https://github.com/huggingface/evaluate/tree/main/metrics/wer) metric.
Before running, make sure to have [ipex-llm](../../../README.md) installed.

## Install Dependencies
```bash
pip install datasets evaluate soundfile librosa jiwer
```

## Run
```bash
export IPEX_LLM_LAST_LM_HEAD=0
python run_whisper.py --model_path /path/to/model --data_type other --device cpu
```

The LibriSpeech dataset contains 'clean' and 'other' splits. 
You can specify the split to evaluate with ```--data_type```.
By default, we set it to ```other```.
You can specify the device to run the test on with  ```--device```.
To run on Intel GPU, set it to ```xpu```, and refer to [GPU installation guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html) for details on installation and optimal configuration.


> **Note**
>
> If you get the error message `ConnectionError: Couldn't reach http://www.openslr.org/resources/12/test-other.tar.gz (error 403)`, you can source from a local dataset instead.

## Using a local dataset
By default, the LibriSpeech dataset is downloaded at runtime with Huggingface Hub. If you prefer to source from a local dataset instead, please set the following environment variable before running the evaluation script

```bash
export LIBRISPEECH_DATASET_PATH=/path/to/dataset_folder
```

Make sure the local dataset folder contains 'dev-other.tar.gz','test-other.tar.gz', and 'train-other-500.tar.gz'. The files can be downloaded from http://www.openslr.org/resources/12/

## Printed metrics
Three metrics are printed:
- Realtime Factor(RTF): RTF indicates total prediction time over the total duration of speech samples.
- Realtime X(RTX): RTX is the inverse of RTF
- Word Error Rate (WER): WER indicates the average number of errors per reference word.