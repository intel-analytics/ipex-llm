# Orca Ray Pong example

We demonstrate how to easily run [mutil-agent](https://github.com/ray-project/ray/blob/master/rllib/examples/multiagent_two_trainers.py)
example provided by [Ray](https://github.com/ray-project/ray). See [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/) for more details for RayOnSpark support in Analytics Zoo.

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install gym
pip install tensorflow
pip install analytics-zoo[ray]
```

## Run example
You can run this example on local mode and yarn client mode. 

- Run with Spark Local mode:
```bash
python multiagent_two_trainers.py
```

- Run with Yarn Client mode, export env `HADOOP_CONF_DIR`:
```bash
python multiagent_two_trainers.py --cluster_mode yarn
```

In above commands
* `--cluster_mode` The mode of spark cluster, supporting local and yarn. Default is "local".
- `--object_store_memory`The store memory you need to use on local. Default is 4g.
- `--driver_cores` The number of driver's or local's cpu cores you want to use. Default is 8.
- `--iterations` The number of iterations to train the model. Default is 10.

**Options for yarn only**
- `--slave_num` The number of slave nodes you want to to use. Default is 2.
- `--executor_cores` The number of slave(executor)'s cpu cores you want to use. Default is 8.
- `--executor_memory` The size of slave(executor)'s memory you want to use. Default is 10g.
- `--driver_memory` The size of driver's memory you want to use. Default is 2g
- `--extra_executor_memory_for_ray` The size of slave(executor)'s extra memory to store data. Default is 20g.


## Results
You can find the logs for training:
```
== Iteration 8 ==
-- DQN --
custom_metrics: {}
date: 2021-02-25_13-15-56
done: false
episode_len_mean: 87.37
episode_reward_max: 800.0
episode_reward_mean: 237.78
episode_reward_min: 59.0
episodes_this_iter: 5
episodes_total: 106
experiment_id: 9efe0d8c130e470d8ac8a54e5099fea0
hostname: intern03
info:
  exploration_infos:
  - cur_epsilon: 0.3140000104904175
  grad_time_ms: 6.585
  learner:
    dqn_policy:
      cur_lr: 0.0005000000237487257
      max_q: 14.558453559875488
      mean_q: 12.489272117614746
      mean_td_error: 0.5722663998603821
      min_q: 1.586671233177185
      model: {}
  num_steps_sampled: 9000
  num_steps_trained: 64000
  num_target_updates: 17
  opt_peak_throughput: 4859.264
  opt_samples: 32.0
  replay_time_ms: 3.196
  sample_time_ms: 16.764
  update_time_ms: 0.004
iterations_since_restore: 9
node_ip: 10.239.166.24
num_healthy_workers: 0
off_policy_estimator: {}
perf:
  cpu_util_percent: 28.30952380952381
  ram_util_percent: 16.300000000000004
pid: 21993
policy_reward_max:
  dqn_policy: 200.0
  ppo_policy: 200.0
policy_reward_mean:
  dqn_policy: 57.22
  ppo_policy: 61.67
policy_reward_min:
  dqn_policy: 8.0
  ppo_policy: 9.0
sampler_perf:
  mean_env_wait_ms: 0.10796930234134994
  mean_inference_ms: 2.0441268517030435
  mean_processing_ms: 0.8424553870851053
time_since_restore: 51.16335201263428
time_this_iter_s: 6.669646501541138
time_total_s: 51.16335201263428
timestamp: 1614230156
timesteps_since_restore: 9000
timesteps_this_iter: 1000
timesteps_total: 9000
training_iteration: 9

-- PPO --
custom_metrics: {}
date: 2021-02-25_13-16-05
done: false
episode_len_mean: 199.95
episode_reward_max: 800.0
episode_reward_mean: 609.93
episode_reward_min: 391.0
episodes_this_iter: 20
episodes_total: 327
experiment_id: 627dde9582a64ee78d15793bfb963610
hostname: intern03
info:
  grad_time_ms: 4742.87
  learner:
    ppo_policy:
      cur_kl_coeff: 0.02812499925494194
      cur_lr: 4.999999873689376e-05
      entropy: 0.5380439162254333
      entropy_coeff: 0.0
      kl: 0.006588105112314224
      model: {}
      policy_loss: -0.00467092776671052
      total_loss: 245.4273681640625
      vf_explained_var: 0.6153221130371094
      vf_loss: 245.43186950683594
  load_time_ms: 5.567
  num_steps_sampled: 36000
  num_steps_trained: 62848
  sample_time_ms: 2830.304
  update_time_ms: 101.258
iterations_since_restore: 9
node_ip: 10.239.166.24
num_healthy_workers: 2
off_policy_estimator: {}
perf:
  cpu_util_percent: 27.922727272727272
  ram_util_percent: 16.300000000000004
pid: 21993
policy_reward_max:
  dqn_policy: 200.0
  ppo_policy: 200.0
policy_reward_mean:
  dqn_policy: 108.475
  ppo_policy: 196.49
policy_reward_min:
  dqn_policy: 14.0
  ppo_policy: 23.0
sampler_perf:
  mean_env_wait_ms: 0.08530574958488021
  mean_inference_ms: 1.0052360749570775
  mean_processing_ms: 0.23539870662498744
time_since_restore: 69.19610691070557
time_this_iter_s: 8.62504243850708
time_total_s: 69.19610691070557
timestamp: 1614230165
timesteps_since_restore: 36000
timesteps_this_iter: 4000
timesteps_total: 36000
training_iteration: 9
```
