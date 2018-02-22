import numpy as np
import scipy.signal
from bigdl.util.common import JTensor, Sample
from bigdl.models.utils.model_broadcast import broadcast_model


def _discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def discounted_sum_of_future_rewards(rewards, gamma):
    result = _discount(rewards, gamma)
    return result


def normalize(advantages, small_eps=1e-8):
    return (advantages - advantages.mean())/(advantages.std() + small_eps)


def obs_act_adv_to_sample(x, sparse=True):
    if sparse:
        return _to_sample_sparse(x)
    else:
        return _to_sample_dense(x)

def _to_sample_sparse(x):
    input_tensor = JTensor.from_ndarray(x[0]) #obs
    values = np.array([x[2]]) # adv
    indices = np.array([x[1]]) # act
    shape = np.array([2])
    target_tensor = JTensor.sparse(values, indices, shape, bigdl_type="float")
    return Sample.from_jtensor(input_tensor, target_tensor)

def _to_sample_dense(x):
    label = np.zeros(2)
    label[x[1]] = x[2]
    return Sample.from_ndarray(x[0], label)


class SampledTrajs(object):
    def __init__(self, sc, agents, policy, num_steps_per_part=None, num_trajs_per_part=None):
        self.agents = agents
        self.policy = policy
        self.sc = sc
        self.num_steps_per_part = num_steps_per_part
        self.num_trajs_per_part = num_trajs_per_part

    def __enter__(self):
        broadcasted = broadcast_model(self.sc, self.policy)
        steps = self.num_steps_per_part
        trajs = self.num_trajs_per_part
        self.samples = self.agents\
            .flatMap(lambda agent: agent.sample(broadcasted.value, num_steps=steps, num_trajs=trajs)).cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.samples.unpersist()

