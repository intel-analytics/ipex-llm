from .optimizer import InferenceOptimizer
from bigdl.nano.utils.inference.common.utils import AccelerationOption

class ChannelsLastOption(AccelerationOption):
    def __init__(self):
        super().__init__(channels_last=True)
    
    def optimize(model, training_data=None, input_sample=None,
                 thread_num=None, logging=False, sample_size_for_pot=100):
        