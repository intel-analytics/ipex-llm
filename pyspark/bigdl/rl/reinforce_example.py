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
from bigdl.rl.agent import *
from bigdl.rl.utils import *
import gym
from gym import wrappers
import math

GAMMA = 0.95


def build_model(state_size):
    model = Sequential()

    model.add(Linear(state_size, 10))
    model.add(Tanh())

    model.add(Linear(10, 2))
    model.add(SoftMax())
    return model


def create_agent(x):
    env = gym.make('CartPole-v1')
    return REINFORCEAgent(env, 498)


def calc_baseline(r_rewards):
    max_steps = r_rewards.map(lambda x: x.shape[0]).max()
    pad = r_rewards.map(lambda x: np.pad(x, (0, max_steps-x.shape[0]), 'constant'))
    sum, count = pad.map(lambda x: (x, 1)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    mean = sum / count
    return mean


def normalize(records, eps=1e-8):
    stats = records.map(lambda x: x[2]).stats()

    mean = stats.mean()
    std = stats.sampleStdev()

    return records.map(lambda x: (x[0], x[1], (x[2] - mean) / (std + eps)))


if __name__ == "__main__":
    spark_conf = create_spark_conf()

    sc = SparkContext(appName="REINFORCE_CartPole-v1", conf=spark_conf)
    init_engine()
    init_executor_gateway(sc)
    redire_spark_logs()
    # show_bigdl_logs("DEBUG")

    node_num, core_num = get_node_and_core_number()
    parallelism = node_num * core_num

    print "parallelism %s " % parallelism
    #
    env = gym.make('CartPole-v1')
    env = wrappers.Monitor(env, "/tmp/cartpole-experiment", video_callable=lambda x: True, force=True)
    test_agent = REINFORCEAgent(env, 1000)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = build_model(state_size)
    criterion = PGCriterion()

    with DistributedAgents(sc, create_agent=create_agent, parallelism=parallelism) as a:
        agents = a.agents
        optimizer = None
        num_trajs_per_part = int(math.ceil(15.0 / parallelism))

        for i in range(60):
            with SampledTrajs(sc, agents, model, num_trajs_per_part=num_trajs_per_part) as trajs:
                trajs = trajs.samples \
                    .map(lambda traj: (traj.data["observations"],
                                       traj.data["actions"],
                                       traj.data["rewards"]))

                rewards_stat = trajs.map(lambda traj: traj[2].sum()).stats()

                print "*********** steps %s **************" % i
                print "reward mean:", rewards_stat.mean()
                print "reward std:", rewards_stat.sampleStdev()
                print "reward max:", rewards_stat.max()

                # calculate the discounted sum of future rewards
                trajs = trajs.map(lambda x: (x[0], x[1], discounted_sum_of_future_rewards(x[2], GAMMA)))

                # calculate advantages
                baseline = calc_baseline(trajs.map(lambda x: x[2]))
                trajs = trajs.map(lambda x: (x[0], x[1], x[2] - baseline[:len(x[2])]))

                # trajectories to records
                records = trajs.flatMap(lambda x: [(x[0][i], x[1][i], x[2][i]) for i in range(len(x[0]))])

                num_records = records.count()
                batch_size = num_records - num_records % parallelism

                print "total %s num_records" % num_records
                print "using %s batch_size" % batch_size

                # normalize advantages
                normalized = normalize(records)

                # to bigdl sample
                data = normalized.map(obs_act_adv_to_sample)

                # update one step
                if optimizer is None:
                    optimizer = Optimizer(model=model,
                                          training_rdd=data,
                                          criterion=criterion,
                                          optim_method=RMSprop(learningrate=0.005),
                                          end_trigger=MaxIteration(1),
                                          batch_size=batch_size)
                else:
                    optimizer.set_traindata(data, batch_size)
                    optimizer.set_end_when(MaxIteration(i + 1))

                model = optimizer.optimize()

                if (i + 1) % 10 == 0:
                    step = test_agent.sample(model, num_trajs=1)[0].data["actions"].shape[0]
                    print "************************************************************************"
                    print "*****************sample video generated, %s steps**********************" % step
                    print "************************************************************************"

    env.close()

