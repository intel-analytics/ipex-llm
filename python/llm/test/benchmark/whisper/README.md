# Whisper Test
The Whisper Test allows users to evaluate the performance and accuracy of Whisper speech-to-text models.
For accuracy, the model is tested on the [LibriSpeech] (https://huggingface.co/datasets/librispeech_asr) dataset using [Word Error Rate (WER)](https://github.com/huggingface/evaluate/tree/main/metrics/wer) metric.
Before running, make sure to have [bigdl-llm](../../../README.md) installed.

## Install Dependencies
```bash
pip install datasets evaluate soundfile librosa jiwer
```

## Run
The LibriSpeech dataset contains 'clean' and 'other' splits. By default, the evaluation is done on the 'other' split.

### Evaluation on CPU
```bash
python run_whisper.py --model_path /path/to/model --data_type other --device cpu
```

### Evaluation on GPU
```bash
python run_whisper.py --model_path /path/to/model --data_type other --device xpu
```

> **Note**
>
> If you get the error message `ConnectionError: Couldn't reach http://www.openslr.org/resources/12/test-other.tar.gz (error 403)`, you can source from a local dataset instead.

## Using a local dataset
By default, the LibriSpeech dataset is downloaded at runtime with Huggingface Hub. If you prefer to source from a local dataset instead, please set the following environment variable before running the evaluation script

```bash
export LIBRISPEECH_DATASET_PATH=/path/to/dataset_folder
```

Make sure the local dataset folder contains 'dev-other.tar.gz','test-other.tar.gz', and 'train-other-500.tar.gz'. The files can be downloaded from http://www.openslr.org/resources/12/
