#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.rl.agent import REINFORCEAgent
import gym
from bigdl.models.utils.model_broadcast import broadcast_model


def build_model(state_size):
    model = Sequential()

    model.add(Linear(state_size, 10))
    model.add(Tanh())

    model.add(Linear(10, 2))
    model.add(SoftMax())
    return model


def to_sample(x):
    input_tensor = JTensor.from_ndarray(x[0]) #obs
    values = np.array([x[2]]) # adv
    indices = np.array([x[1]]) # act
    shape = np.array([2])
    target_tensor = JTensor.sparse(values, indices, shape, bigdl_type="float")
    return Sample.from_jtensor(input_tensor, target_tensor)

def to_sample_dense(x):
    input_tensor = JTensor.from_ndarray(x[0]) #obs
    values = np.array([x[2]]) # adv
    indices = np.array([x[1]]) # act
    shape = np.array([2])
    target_tensor = JTensor.sparse(values, indices, shape, bigdl_type="float")
    label = np.zeros(2)
    label[x[1]] = x[2]
    return Sample.from_ndarray(x[0], label)


def createAgent(x):
    env = gym.make('CartPole-v1')
    return REINFORCEAgent(env, 498)


if __name__ == "__main__":
    spark_conf = create_spark_conf()

    sc = SparkContext(appName="lenet5", conf=spark_conf)
    sc.setLogLevel("ERROR")
    redire_spark_logs()
    # show_bigdl_info_logs()
    init_engine()
    init_executor_gateway(sc)
    #
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(env, 498)

    agents = sc.parallelize(range(4), 4).map(createAgent).cache()

    model = build_model(state_size)

    optimizer = None

    for i in range(100):
        broadcasted = broadcast_model(sc, model)

        trajs = agents.flatMap(lambda agent: agent.sample(broadcasted.value, 500)).cache()

        samples = trajs.flatMap(lambda x: [(x.data["observations"][i],
                                            x.data["actions"][i],
                                            x.data["rewards"][i]) for i in range(len(x.data["actions"]))])

        stats = samples.map(lambda x: x[2]).stats()

        mean = stats.mean()
        std = stats.sampleStdev()

        normalize_samples = samples.map(lambda x: (x[0], x[1], (x[2] - mean)/std))

        data = normalize_samples.map(to_sample)

        criterion = PGCriterion()

        if optimizer is None:
            optimizer = Optimizer(model=model,
                              training_rdd=data,
                              criterion=criterion,
                              optim_method=RMSprop(learningrate=0.005),
                              end_trigger=MaxIteration(1),
                              batch_size=2000)
        else:
            optimizer.set_traindata(data, 2000)
            optimizer.set_end_when(MaxIteration(i + 1))

        model = optimizer.optimize()

        samples.unpersist()

        step = len(agent.sample(model, num_trajs=1)[0].data)

        print "current num of steps: %s" % step