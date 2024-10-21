import copy

class InferenceOptimizer():

    def __init__(self) -> None:
        self.opt = None
        self.trans_pipeline = None

    def optimize(self, trans_pipeline, training_data, *args, **kwargs):
        if self.opt is None:
            if trans_pipeline.framework == "pt":
                from bigdl.nano.pytorch import InferenceOptimizer as PTOptimizer
                self.opt = PTOptimizer()
            elif trans_pipeline.framework == "tf":
                from bigdl.nano.tf import InferenceOptimizer as TFOptimizer
                self.opt = TFOptimizer()
            else:
                raise ValueError("Unsupported framework: {}".format(trans_pipeline.framework))

        training_data = trans_pipeline.preprocess(training_data)

        self.opt.optimize(trans_pipeline.model, training_data=training_data)
        self.trans_pipeline = trans_pipeline
    
    def get_best_model(self):
        model, options = self.opt.get_best_model()
        p = copy.deepcopy(self.trans_pipeline)
        p.model = model
        return p, options