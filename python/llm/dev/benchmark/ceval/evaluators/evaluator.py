import re
import string
class Evaluator:
    def __init__(self, choices, model_path, device, qtype):
        self.choices = choices
        self.model_path = model_path
        self.device = device
        self.qtype = qtype

    def format_example(self, line, **kwargs):
        pass
    
    def eval_subject(self, subject_name, test_df, eval_type, **kwargs):
        pass

    def extract_answer(self, response, row, **kwargs):
        pass
