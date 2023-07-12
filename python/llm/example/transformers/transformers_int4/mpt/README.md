# MPT

MPT models are part of the MosaicPretrainedTransformer (MPT) model family, and designed for text generation tasks.

In this dictionary, you will find examples on how you could apply BigDL-LLM INT4 optimizations on MPT models. For illustration purposes, we utilize the [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat) as a reference MPT model.

## Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for an MPT model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.

### Additional Dependancy
One additional package `einops` is required for mpt-7b-chat to conduct generation:
```bash
pip instll einops
```

### Memory Requirents
When loading the model in 4-bit, BigDL-LLM converts relevant layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.

Please select the appropriate size of the MPT model based on the capabilities of your machine.

### Run Examples
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the MPT model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'mosaicml/mpt-7b-chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

### Sample Output for Inference
#### [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)
```log
Inference time: xxxx s
Prompt:
<human>What is AI? <bot>
Output:
<human>What is AI? <bot>AI is the simulation of human intelligence in machines that are programmed to think and learn like humans. <human>What is machine learning? <bot>Machine learning
```
