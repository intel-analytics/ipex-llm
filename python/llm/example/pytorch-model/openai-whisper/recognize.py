import whisper
import librosa
import argparse
from bigdl.llm import optimize_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recognize Tokens using `transcribe()` API for Openai Whisper model')
    parser.add_argument('--model-size', type=str, default="tiny",
                        help="one of the official model names listed by `whisper.available_models()`, or"
                             "path to a model checkpoint containing the model dimensions and the model state_dict.")
    parser.add_argument('--audio-file', type=str, required=True,
                        help='The path of the audio file to be recognized.')
    parser.add_argument('--language', type=str, default="English",
                        help='language to be transcribed')
    args = parser.parse_args()

    # Load the input audio
    y, sr = librosa.load(args.audio_file)

    # Downsample the audio to 16kHz
    target_sr = 16000
    audio = librosa.resample(y,
                            orig_sr=sr,
                            target_sr=target_sr)

    # Load whisper model under pytorch framework
    model = whisper.load_model(args.model_size)

    # With only one line to enable bigdl optimize on a pytorch model
    model = optimize_model(model, low_bit="sym_int4", optimize_llm=True)

    result = model.transcribe(audio, verbose=True, language=args.language)

    print(result["text"])
