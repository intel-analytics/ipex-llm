import time
import librosa
import argparse

from transformers import pipeline
from bigdl.llm.transformers import AutoModelForSpeechSeq2Seq
from transformers.models.whisper import WhisperFeatureExtractor, WhisperTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognize Tokens using `generate()` API for Whisper model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="openai/whisper-medium",
                        help='The huggingface repo id for the Whisper model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--audio-file', type=str, required=True,
                        help='The path of the audio file to be recognized.')
    parser.add_argument('--language', type=str, default="english",
                        help='language to be transcribed')
    parser.add_argument('--chunk-length', type=str, default=30,
                        help="The maximum number of chuncks of sampling_rate samples used to trim"
                             "and pad longer or shorter audio sequences.")

    args = parser.parse_args()

    # Path to the .wav audio file
    audio_file_path = args.audio_file
    model_path = args.repo_id_or_model_path

    # Load the input audio
    y, sr = librosa.load(audio_file_path, sr=None)

    # Downsample the audio to 16kHz
    target_sr = 16000
    audio = librosa.resample(y,
                            orig_sr=sr,
                            target_sr=target_sr)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, load_in_4bit=True)
    model.config.forced_decoder_ids = None

    pipe = pipeline(
      "automatic-speech-recognition",
      model=model,
      feature_extractor= WhisperFeatureExtractor.from_pretrained(model_path),
      tokenizer= WhisperTokenizer.from_pretrained(model_path, language=args.language),
      chunk_length_s=args.chunk_length,
    )

    start = time.time()
    prediction = pipe(audio, batch_size=2)["text"]
    print(f"inference time is {time.time()-start}")

    print(prediction)
