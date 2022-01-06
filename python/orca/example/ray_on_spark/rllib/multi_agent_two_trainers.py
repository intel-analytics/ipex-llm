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

from ray.rllib.agents.dqn import DQNTrainer, DQNTFPolicy, DQNTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext

parser = argparse.ArgumentParser()
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The mode for the Spark cluster. local, yarn or spatk-submit.')
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
# Use torch for both policies.
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=10,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=50.0,
    help="Reward at which we stop training.")

if __name__ == "__main__":
    args = parser.parse_args()

    cluster_mode = args.cluster_mode
    if cluster_mode.startswith("yarn"):
        if cluster_mode == "yarn-client":
            sc = init_orca_context(cluster_mode="yarn-client",
                                cores=args.executor_cores,
                                memory=args.executor_memory,
                                init_ray_on_spark=True,
                                driver_memory=args.driver_memory,
                                driver_cores=args.driver_cores,
                                num_executors=args.slave_num,
                                extra_executor_memory_for_ray=args.extra_executor_memory_for_ray,
                                object_store_memory=args.object_store_memory)
        else:
            sc = init_orca_context(cluster_mode="yarn-cluster",
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
    elif cluster_mode == "spark-submit":
        sc = init_orca_context(cluster_mode=cluster_mode)
        ray_ctx = OrcaContext.get_ray_context()
    else:
        print("init_orca_context failed. cluster_mode should be one of 'local', 'yarn' and 'spark-submit' but got "
              + cluster_mode)

    # Simple environment with 4 independent cartpole entities
    register_env("multi_agent_cartpole",
                 lambda _: MultiAgentCartPole({"num_agents": 4}))
    single_dummy_env = gym.make("CartPole-v0")
    obs_space = single_dummy_env.observation_space
    act_space = single_dummy_env.action_space

    # You can also have multiple policies per trainer, but here we just
    # show one each for PPO and DQN.
    policies = {
        "ppo_policy": (PPOTorchPolicy if args.framework == "torch" else
                       PPOTFPolicy, obs_space, act_space, {}),
        "dqn_policy": (DQNTorchPolicy if args.framework == "torch" else
                       DQNTFPolicy, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id % 2 == 0:
            return "ppo_policy"
        else:
            return "dqn_policy"

    ppo_trainer = PPOTrainer(
        env="multi_agent_cartpole",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["ppo_policy"],
            },
            "model": {
                "vf_share_layers": True,
            },
            "num_sgd_iter": 6,
            "vf_loss_coeff": 0.01,
            # disable filters, otherwise we would need to synchronize those
            # as well to the DQN agent
            "observation_filter": "MeanStdFilter",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": args.framework,
        })

    dqn_trainer = DQNTrainer(
        env="multi_agent_cartpole",
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["dqn_policy"],
            },
            "model": {
                "vf_share_layers": True,
            },
            "gamma": 0.95,
            "n_step": 3,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "framework": args.framework,
        })

    # You should see both the printed X and Y approach 200 as this trains:
    # info:
    #   policy_reward_mean:
    #     dqn_policy: X
    #     ppo_policy: Y
    for i in range(args.stop_iters):
        print("== Iteration", i, "==")

        # improve the DQN policy
        print("-- DQN --")
        result_dqn = dqn_trainer.train()
        print(pretty_print(result_dqn))

        # improve the PPO policy
        print("-- PPO --")
        result_ppo = ppo_trainer.train()
        print(pretty_print(result_ppo))

        # Test passed gracefully.
        if args.as_test and \
                result_dqn["episode_reward_mean"] > args.stop_reward and \
                result_ppo["episode_reward_mean"] > args.stop_reward:
            print("test passed (both agents above requested reward)")
            quit(0)

        # swap weights to synchronize
        dqn_trainer.set_weights(ppo_trainer.get_weights(["ppo_policy"]))
        ppo_trainer.set_weights(dqn_trainer.get_weights(["dqn_policy"]))

    # Desired reward not reached.
    if args.as_test:
        raise ValueError("Desired reward ({}) not reached!".format(
            args.stop_reward))
    ray_ctx.stop()
    stop_orca_context()
