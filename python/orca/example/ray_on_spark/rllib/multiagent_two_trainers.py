# This file is adapted from https://github.com/ray-project/ray/blob
# /master/python/ray/rllib/examples/multiagent_two_trainers.py
#
# Copyright 2018 Analytics Zoo Authors.
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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Example of using two different training methods at once in multi-agent.
Here we create a number of CartPole agents, some of which are trained with
DQN, and some of which are trained with PPO. We periodically sync weights
between the two trainers (note that no such syncing is needed when using just
a single training method).
For a simpler example, see also: multiagent_cartpole.py
"""

import argparse
import gym
import os

from ray.rllib.agents.dqn.dqn import DQNTrainer
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.tests.test_multi_agent_env import MultiAgentCartPole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca import OrcaContext

os.environ["LANG"] = "C.UTF-8"

parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local or yarn.')
parser.add_argument("--iterations", type=int, default=10,
                    help="The number of iterations to train the model")
parser.add_argument("--slave_num", type=int, default=2,
                    help="The number of slave nodes")
parser.add_argument("--executor_cores", type=int, default=8,
                    help="The number of driver's cpu cores you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--executor_memory", type=str, default="10g",
                    help="The size of slave(executor)'s memory you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--driver_memory", type=str, default="2g",
                    help="The size of driver's memory you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--driver_cores", type=int, default=8,
                    help="The number of driver's cpu cores you want to use."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--extra_executor_memory_for_ray", type=str, default="20g",
                    help="The extra executor memory to store some data."
                         "You can change it depending on your own cluster setting.")
parser.add_argument("--object_store_memory", type=str, default="4g",
                    help="The memory to store data on local."
                         "You can change it depending on your own cluster setting.")

if __name__ == "__main__":
    args = parser.parse_args()
    cluster_mode = args.cluster_mode
    if cluster_mode == "yarn":
        sc = init_orca_context(cluster_mode="yarn",
                               cores=args.executor_cores,
                               memory=args.executor_memory,
                               init_ray_on_spark=True,
                               driver_memory=args.driver_memory,
                               driver_cores=args.driver_cores,
                               num_executors=args.slave_num,
                               extra_executor_memory_for_ray=args.extra_executor_memory_for_ray,
                               object_store_memory=args.object_store_memory)
        ray_ctx = OrcaContext.get_ray_context()
    elif cluster_mode == "local":
        sc = init_orca_context(cores=args.driver_cores)
        ray_ctx = OrcaContext.get_ray_context()
    else:
        print("init_orca_context failed. cluster_mode should be either 'local' or 'yarn' but got "
              + cluster_mode)

    # Simple environment with 4 independent cartpole entities
    register_env("multi_cartpole", lambda _: MultiAgentCartPole({"num_agents": 4}))
    single_env = gym.make("CartPole-v0")
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # You can also have multiple policies per trainer, but here we just
    # show one each for PPO and DQN.
    policies = {
        "ppo_policy": (PPOTFPolicy, obs_space, act_space, {}),
        "dqn_policy": (DQNTFPolicy, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id):
        if agent_id % 2 == 0:
            return "ppo_policy"
        else:
            return "dqn_policy"

    ppo_trainer = PPOTrainer(
        env="multi_cartpole",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["ppo_policy"],
            },
            # disable filters, otherwise we would need to synchronize those
            # as well to the DQN agent
            "observation_filter": "NoFilter",
        })

    dqn_trainer = DQNTrainer(
        env="multi_cartpole",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["dqn_policy"],
            },
            "gamma": 0.95,
            "n_step": 3,
        })

    # You should see both the printed X and Y approach 200 as this trains:
    # info:
    #   policy_reward_mean:
    #     dqn_policy: X
    #     ppo_policy: Y
    for i in range(args.iterations):
        print("== Iteration", i, "==")

        # improve the DQN policy
        print("-- DQN --")
        print(pretty_print(dqn_trainer.train()))

        # improve the PPO policy
        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        # swap weights to synchronize
        dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
        ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))

    ray_ctx.stop()
    stop_orca_context()
