import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("wer")
launch_gradio_widget(module)
