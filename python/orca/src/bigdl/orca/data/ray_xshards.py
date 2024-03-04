#
# Copyright 2016 The BigDL Authors.
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
from __future__ import annotations
from collections import defaultdict

import ray
import ray._private.services
import uuid
import random

from bigdl.orca.data import XShards
from bigdl.orca.ray import OrcaRayContext

import logging
from bigdl.dllib.utils.log4Error import invalidInputError

from numpy import ndarray
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Tuple,
    Union,
    no_type_check
)


if TYPE_CHECKING:
    from bigdl.orca.data.shard import SparkXShards
    from pyspark.rdd import PipelinedRDD, RDD
    from ray._raylet import ObjectRef
    from ray.actor import ActorHandle


logger = logging.getLogger(__name__)


class LocalStore:

    def __init__(self):
        self.partitions = {}

    def upload_shards(self, part_shard_id, shard_ref_list):
        partition_idx, shard_idx = part_shard_id
        if partition_idx not in self.partitions:
            self.partitions[partition_idx] = {}
        shard_ref = shard_ref_list[0]
        self.partitions[partition_idx][shard_idx] = shard_ref
        return 0

    def upload_partition(self, partition_id, partition_ref_list):
        self.partitions[partition_id] = partition_ref_list[0]
        return 0

    def get_shards(self, part_shard_id):
        partition_idx, shard_idx = part_shard_id
        return self.partitions[part_shard_id][shard_idx]

    def get_partition_ref(self, partition_id):
        """
        return a list of shard_refs or a part_ref
        """
        part = self.partitions[partition_id]
        if isinstance(part, dict):
            partition = []
            for shard_idx in range(len(part)):
                shard = part[shard_idx]
                partition.append(shard)
            return partition
        else:
            return part

    def get_partition(self, partition_id):
        """
        return partition_data
        """
        return ray.get(self.get_partition_ref(partition_id))

    def get_partitions_refs(self):
        """
        return a dictionary of partitions, each partition is a list of shard_refs or a part_ref
        """
        result = {}
        for k in self.partitions.keys():
            result[k] = self.get_partition_ref(k)
        return result


def init_ray_if_not(redis_address, redis_password):
    if not ray.is_initialized():
        init_params = dict(
            address=redis_address,
            ignore_reinit_error=True,
            namespace="bigdl"
        )
        if redis_password:
            init_params["_redis_password"] = redis_password
        ray.init(**init_params)


def write_to_ray(idx, partition, redis_address, redis_password, partition_store_names):
    import pandas
    init_ray_if_not(redis_address, redis_password)
    ip = ray._private.services.get_node_ip_address()
    local_store_name = None
    for name in partition_store_names:
        if name.endswith(ip):
            local_store_name = name
            break
    if local_store_name is None:
        local_store_name = random.choice(partition_store_names)

    local_store = ray.get_actor(local_store_name)

    # directly calling ray.put will set this driver as the owner of this object,
    # when the spark job finished, the driver might exit and make the object
    # eligible for deletion.
    result = []
    for shard_id, shard in enumerate(partition):
        shard_ref = ray.put(shard, _owner=local_store)
        result.append(local_store.upload_shards.remote((idx, shard_id), [shard_ref]))

    is_empty = len(result) == 0
    if is_empty:
        partition_ref = ray.put([], _owner=local_store)
        result.append(local_store.upload_partition.remote(idx, [partition_ref]))
        logger.warning(f"Partition {idx} is empty.")
    ray.get(result)

    return [(idx, local_store_name.split(":")[-1], local_store_name)]


def get_from_ray(idx, redis_address, redis_password, idx_to_store_name):
    init_ray_if_not(redis_address, redis_password)
    local_store_handle = ray.get_actor(idx_to_store_name[idx])
    partition = ray.get(local_store_handle.get_partition.remote(idx))
    return partition


class RayXShards(XShards):

    def __init__(self, uuid: str, id_ip_store_rdd: "RDD[Any]",
                 partition_stores: Dict[str, "ActorHandle"]) -> None:
        self.uuid = uuid
        self.rdd = id_ip_store_rdd
        self.partition_stores = partition_stores
        self.id_ip_store = self.rdd.collect()
        self.partition2store_name = {idx: store_name for idx, _, store_name in self.id_ip_store}
        self.partition2ip = {idx: ip for idx, ip, _ in self.id_ip_store}

    def transform_shard(self, func, *args):
        invalidInputError(False, "Transform is not supported for RayXShards")

    def num_partitions(self) -> int:
        return len(self.partition2ip)

    def collect(self) -> List[Dict[str, ndarray]]:
        # return a list of shards
        partitions = self.collect_partitions()
        data = [item for part in partitions for item in part]
        return data

    def get_partition_refs(self) -> List[Union["ObjectRef", List["ObjectRef"]]]:
        """
        Get a list of partition_refs, each partition_ref is a list of shard_refs or a partition_ref
        """
        # part_shard_refs is a list of partitions, each partition is a dictionary,
        # with key of partition index and value of (list of shard_refs or part_ref)
        part_shard_refs = ray.get([local_store.get_partitions_refs.remote()
                                   for local_store in self.partition_stores.values()])
        result = {}
        for part in part_shard_refs:
            result.update(part)
        return [result[idx] for idx in range(self.num_partitions())]

    def get_refs(self) -> List["ObjectRef"]:
        """
        Flatten get_partition_refs. Get a list of partition_refs or shard_refs
        """
        partition_refs = self.get_partition_refs()
        return [ref for partition_ref in partition_refs for ref in partition_ref]  # type:ignore

    def collect_partitions(self) -> List[List[Dict[str, ndarray]]]:
        part_refs = self.get_partition_refs()
        return [ray.get(part_ref) for part_ref in part_refs]

    def to_spark_xshards(self) -> "SparkXShards":
        from bigdl.orca.data import SparkXShards
        ray_ctx = OrcaRayContext.get()
        address = ray_ctx.redis_address  # type: ignore
        password = ray_ctx.redis_password  # type: ignore
        partition2store = self.partition2store_name
        rdd = self.rdd.mapPartitionsWithIndex(
            lambda idx, _: get_from_ray(idx, address, password, partition2store))

        # the reason why we trigger computation here is to ensure we get the data
        # from ray before the RayXShards goes out of scope and the data get garbage collected
        rdd = rdd.cache()
        result_rdd = rdd.map(lambda x: x)  # sparkxshards will uncache the rdd when gc
        spark_xshards = SparkXShards(result_rdd)
        return spark_xshards

    def _get_multiple_partition_refs(self, ids: List[int]) -> List["ObjectRef"]:
        refs = []
        for idx in ids:
            local_store_handle = self.partition_stores[self.partition2store_name[idx]]
            partition_ref = local_store_handle.get_partition.remote(idx)
            refs.append(partition_ref)
        return refs

    def transform_shards_with_actors(self, actors: List["ActorHandle"],
                                     func: Callable) -> "RayXShards":
        """
        Assign each partition_ref (referencing a list of shards) to an actor,
        and run func for each actor and partition_ref pair.
        Actors should have a `get_node_ip` method to achieve locality scheduling.
        The `get_node_ip` method should call ray._private.services.get_node_ip_address()
        to return the correct ip address.
        The `func` should take an actor and a partition_ref as argument and
        invoke some remote func on that actor and return a new partition_ref.
        Note that if you pass partition_ref directly to actor method, ray
        will resolve that partition_ref to the actual partition object, which
        is a list of shards. If you pass partition_ref indirectly through other
        object, say [partition_ref], ray will send the partition_ref itself to
        actor, and you may need to use ray.get(partition_ref) on actor to retrieve
        the actor partition objects.
        """
        assigned_partitions, actor_ips, assigned_actors = self.assign_partitions_to_actors(actors)
        assigned_partition_refs = [(part_ids, self._get_multiple_partition_refs(part_ids))
                                   for part_ids in assigned_partitions]
        new_part_id_refs = {
            part_id: func(actor, part_ref)
            for actor, (part_ids, part_refs) in zip(assigned_actors, assigned_partition_refs)
            for part_id, part_ref in zip(part_ids, part_refs)}

        actor_ip2part_id = defaultdict(list)
        for actor_ip, part_ids in zip(actor_ips, assigned_partitions):
            actor_ip2part_id[actor_ip].extend(part_ids)

        return RayXShards.from_partition_refs(actor_ip2part_id, new_part_id_refs, self.rdd)

    # TODO: ADD UT
    def reduce_partitions_for_actors(self, actors: List["ActorHandle"],
                                     reduce_partitions_func: Callable,
                                     return_refs: bool=False) -> List[Any]:
        """
        Evenly allocate partitions for actors and run `reduce_partitions_func` on partitions of each
        worker.
        Return a list of results, where one result corresponds to one worker.

        :param actors: ray actors
        :param reduce_partitions_func: Function to run on each ray actor which reduces the
            partition refs on the actor to one result_ref. It should take an actor and a list of
            partition_refs as argument return a result_ref
        :param return_refs: Whether to return ray objects refs or ray objects. If True, return a
        list of ray object refs, otherwise return a list of ray objects. Defaults to be False,
        """
        invalidInputError(self.num_partitions() >= len(actors),
                          f"Get the number of partitions ({self.num_partitions()}) smaller than "
                          f"the number of workers ({len(actors)}).")
        assigned_partitions, _, _ = self.assign_partitions_to_actors(actors)
        result_refs = []
        for actor, part_ids in zip(actors, assigned_partitions):
            assigned_partition_refs = self._get_multiple_partition_refs(part_ids)
            result_ref = reduce_partitions_func(actor, assigned_partition_refs)
            result_refs.append(result_ref)
        if return_refs:
            return result_refs
        results = ray.get(result_refs)
        return results

    # TODO: ADD UT
    def zip_reduce_shards_with_actors(self,
                                      xshards: "RayXShards",
                                      actors: List["ActorHandle"],
                                      reduce_partitions_func: Callable,
                                      return_refs: bool=False) -> List[Any]:
        invalidInputError(self.num_partitions() == xshards.num_partitions(),
                          "the rdds to be zipped must have the same number of partitions")
        invalidInputError(self.num_partitions() >= len(actors),
                          f"Get number of partitions ({self.num_partitions()}) smaller than"
                          f" number of actors ({len(actors)}). Please submit an issue"
                          f" to BigDL.")
        assigned_partitions, _, _ = self.assign_partitions_to_actors(actors)
        result_refs = []
        for actor, part_ids in zip(actors, assigned_partitions):
            assigned_partition_refs = self._get_multiple_partition_refs(part_ids)
            assigned_partition_refs_other = xshards._get_multiple_partition_refs(part_ids)
            result_ref = reduce_partitions_func(actor, assigned_partition_refs,
                                                assigned_partition_refs_other)
            result_refs.append(result_ref)
        if return_refs:
            return result_refs
        results = ray.get(result_refs)
        return results

    def assign_partitions_to_actors(self, actors: List["ActorHandle"]) \
            -> Tuple[List[List[int]], List[str], List["ActorHandle"]]:
        num_parts = self.num_partitions()
        if num_parts < len(actors):
            logger.warning(f"this rdd has {num_parts} partitions, which is smaller "
                           f"than actor number ({len(actors)} actors). That could cause "
                           f"unbalancing workload on different actors. We recommend you to "
                           f"repartition the rdd for better performance.")

        avg_part_num = num_parts // len(actors)
        remainder = num_parts % len(actors)

        part_id2ip = self.partition2ip.copy()
        # the assigning algorithm
        # 1. calculate the average partition number per actor avg_part_num, and the number
        #    of remaining partitions remainder. So there are remainder number of actors got
        #    avg_part_num + 1 partitions and other actors got avg_part_num partitions.
        # 2. loop partitions and assign each according to ip, if no actor with this ip or
        #    all actors with this ip have been full, this round of assignment failed.
        # 3. assign the partitions that failed to be assigned to actors that has full

        # todo extract this algorithm to other functions for unit tests.
        actor_ips = []
        for actor in actors:
            invalidInputError(hasattr(actor, "get_node_ip"),  # type:ignore
                              "each actor should have a get_node_ip method")
            actor_ip = actor.get_node_ip.remote()
            actor_ips.append(actor_ip)

        actor_ips = ray.get(actor_ips)

        actor2assignments = [[] for i in range(len(actors))]  # type:ignore

        ip2actors = {}  # type:ignore
        for idx, ip in enumerate(actor_ips):
            if ip not in ip2actors:
                ip2actors[ip] = []
            ip2actors[ip].append(idx)

        unassigned = []
        for part_idx, ip in part_id2ip.items():
            assigned = False
            if ip in ip2actors:
                ip_actors = ip2actors[ip]

                for actor_id in ip_actors:
                    current_assignments = actor2assignments[actor_id]
                    if len(current_assignments) < avg_part_num:
                        current_assignments.append(part_idx)
                        assigned = True
                        break
                    elif len(current_assignments) == avg_part_num and remainder > 0:
                        current_assignments.append(part_idx)
                        remainder -= 1
                        assigned = True
                        break
            if not assigned:
                unassigned.append((part_idx, ip))

        for part_idx, ip in unassigned:
            for current_assignments in actor2assignments:
                if len(current_assignments) < avg_part_num:
                    current_assignments.append(part_idx)
                    break
                elif len(current_assignments) == avg_part_num and remainder > 0:
                    current_assignments.append(part_idx)
                    remainder -= 1
                    break

        if num_parts < len(actors):
            # filter assigned actors
            assigned_actors = []
            assigned_actor2assignments = []
            assigned_actor_ips = []
            for actor, assignment, ip in zip(actors, actor2assignments, actor_ips):
                if assignment:
                    assigned_actors.append(actor)
                    assigned_actor2assignments.append(assignment)
                    assigned_actor_ips.append(ip)
            return assigned_actor2assignments, assigned_actor_ips, assigned_actors
        else:
            return actor2assignments, actor_ips, actors

    @no_type_check
    @staticmethod
    def from_partition_refs(ip2part_id: DefaultDict[str, List[int]],
                            part_id2ref: Dict[int, "ObjectRef"],
                            old_rdd: "RDD[Any]") -> "RayXShards":
        uuid_str = str(uuid.uuid4())
        id2store_name = {}
        partition_stores = {}
        part_id2ip = {}
        result = []
        for node, part_ids in ip2part_id.items():
            name = f"partition:{uuid_str}:{node}"
            store = ray.remote(num_cpus=0, resources={f"node:{node}": 1e-4})(LocalStore) \
                .options(name=name).remote()
            partition_stores[name] = store
            for idx in part_ids:
                result.append(store.upload_partition.remote(idx, [part_id2ref[idx]]))
                id2store_name[idx] = name
                part_id2ip[idx] = node
        ray.get(result)
        new_id_ip_store_rdd = old_rdd.mapPartitionsWithIndex(
            lambda idx, _: [(idx, part_id2ip[idx], id2store_name[idx])]).cache()
        return RayXShards(uuid_str, new_id_ip_store_rdd, partition_stores)

    @staticmethod
    def from_spark_xshards(spark_xshards: "SparkXShards") -> "RayXShards":
        return RayXShards._from_spark_xshards_ray_api(spark_xshards)

    @no_type_check
    @staticmethod
    def _from_spark_xshards_ray_api(spark_xshards: "SparkXShards") -> "RayXShards":
        ray_ctx = OrcaRayContext.get()
        address = ray_ctx.redis_address
        password = ray_ctx.redis_password
        driver_ip = ray._private.services.get_node_ip_address()
        uuid_str = str(uuid.uuid4())
        resources = ray.cluster_resources()
        nodes = []
        for key, value in resources.items():
            if key.startswith("node:"):
                # if running in cluster, filter out driver ip
                if key != f"node:{driver_ip}":
                    nodes.append(key)
        # for the case of local mode and single node spark standalone
        if not nodes:
            nodes.append(f"node:{driver_ip}")

        partition_stores = {}
        for node in nodes:
            name = f"partition:{uuid_str}:{node}"
            store = ray.remote(num_cpus=0, resources={node: 1e-4})(LocalStore)\
                .options(name=name, lifetime="detached").remote()
            partition_stores[name] = store

        # actor creation is async, this is to make sure they all have been started
        ray.get([v.get_partitions_refs.remote() for v in partition_stores.values()])
        partition_store_names = list(partition_stores.keys())
        result_rdd = spark_xshards.rdd.mapPartitionsWithIndex(lambda idx, part: write_to_ray(
            idx, part, address, password, partition_store_names)).cache()

        return RayXShards(uuid_str, result_rdd, partition_stores)
