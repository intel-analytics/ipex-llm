# Whisper Evaluation
The Whisper Evaluation allows users to evaluate the accuracy of Whisper speech-to-text models.
Specifically, the model is tested on the LibriSpeech dataset using Word Error Rate (WER) metric.
Before running, make sure to have [bigdl-llm](../../../README.md) installed.

## Install Dependencies
```bash
pip install datasets
pip install evaluate
pip install soundfile
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
export LIRBISPEECH_DATASET_PATH=/path/to/dataset_folder
```

Make sure the local dataset folder contains 'dev-other.tar.gz','test-other.tar.gz', and 'train-other-500.tar.gz'. The files can be downloaded from http://www.openslr.org/resources/12/
