from . import optimizer
from bigdl.nano.utils.inference.common.utils import AccelerationOption


class ChannelsLastOption(AccelerationOption):
    def __init__(self):
        super().__init__(channels_last=True)
    
    def optimize(self, model, training_data=None, input_sample=None,
                 thread_num=None, logging=False, sample_size_for_pot=100):
        acce_model = optimizer.InferenceOptimizer.trace(model=model,
                                                        channels_last=True)
        return acce_model


class IpexOption(AccelerationOption):
    def __init__(self):
        super().__init__(ipex=True)
    
    def optimize(self, model, training_data=None, input_sample=None,
                 thread_num=None, logging=False, sample_size_for_pot=100):
        acce_model = optimizer.InferenceOptimizer.trace(model=model,
                                                        use_ipex=True)
        return acce_model

