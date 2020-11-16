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
#

import ray
import ray.services
import uuid
import random

from zoo.ray import RayContext


@ray.remote(num_cpus=0)
class MetaStore:

    def __init__(self):
        self.partitions = {}

    def set_partition_ref(self, idx, object_ref):
        self.partitions[idx] = object_ref[0]
        return 0

    def get_partition_ref(self, idx):
        return self.partitions[idx]

    def get_multiple_partition_refs(self, idxs):
        return [self.partitions[idx] for idx in idxs]

    def get_all_partition_refs(self):
        return [self.partitions[i] for i in range(len(self.partitions))]

    def num_partitions(self):
        return len(self.partitions)


class PartitionUploader:

    def __init__(self, meta_store_handle):
        self.meta_store_handle = meta_store_handle

    def upload_partition(self, idx, partition):
        partition_ref = ray.put(partition)
        ray.get(self.meta_store_handle.set_partition_ref.remote(idx, [partition_ref]))
        return 0

    def upload_multiple_partitions(self, idxs, partitions):
        for idx, partition in zip(idxs, partitions):
            self.upload_partition(idx, partition)
        return 0

    def upload_partitions_from_plasma(self, partition_id, plasma_object_id, object_store_address):
        import pyarrow.plasma as plasma
        client = plasma.connect(object_store_address)
        partition = client.get(plasma_object_id)
        partition_ref = ray.put(partition)
        ray.get(self.meta_store_handle.set_partition_ref(partition_id, [partition_ref]))
        return 0


def write_to_ray(idx, partition, redis_address, redis_password, partition_store_names):
    partition = list(partition)
    ray.init(address=redis_address, redis_password=redis_password, ignore_reinit_error=True)
    ip = ray.services.get_node_ip_address()
    local_store_name = None
    for name in partition_store_names:
        if name.endswith(ip):
            local_store_name = name
            break
    if local_store_name is None:
        local_store_name = random.choice(partition_store_names)

    local_store = ray.util.get_actor(local_store_name)

    # directly calling ray.put will set this driver as the owner of this object,
    # when the spark job finished, the driver might exit and make the object
    # eligible for deletion.
    ray.get(local_store.upload_partition.remote(idx, partition))
    ray.shutdown()

    return [(idx, local_store_name.split(":")[-1])]


def get_from_ray(idx, redis_address, redis_password, meta_store_name):
    ray.init(address=redis_address, redis_password=redis_password, ignore_reinit_error=True)
    meta_store_handle = ray.util.get_actor(meta_store_name)
    object_id = ray.get(meta_store_handle.get_partition_ref.remote(idx))
    partition = ray.get(object_id)
    ray.shutdown()
    return partition


class RayRdd:

    def __init__(self, uuid, meta_store, partition2ip):
        self.uuid = uuid
        self.meta_store = meta_store
        self.partition2ip = partition2ip

    def num_partitions(self):
        return ray.get(self.meta_store.num_partitions.remote())

    def collect(self):
        partitions = self.collect_partitions()
        data = [item for part in partitions for item in part]
        return data

    def collect_partitions(self):
        partition_refs = ray.get(self.meta_store.get_all_partition_refs.remote())
        partitions = ray.get(partition_refs)
        return partitions

    def to_spark_rdd(self):
        ray_ctx = RayContext.get()
        sc = ray_ctx.sc
        address = ray_ctx.redis_address
        password = ray_ctx.redis_password
        num_parts = ray.get(self.meta_store.num_partitions.remote())
        meta_store_name = f"meta_store:{self.uuid}"
        rdd = sc.parallelize([0] * num_parts * 10, num_parts)\
            .mapPartitionsWithIndex(
            lambda idx, _: get_from_ray(idx, address, password, meta_store_name))
        return rdd

    def _get_multiple_partition_refs(self, ids):
        result = ray.get(self.meta_store.get_multiple_partition_refs.remote(ids))
        return result

    def map_partitions_with_actors(self, actors, func, gang_scheduling=True):
        assigned_partitions, actor_ips = self.assign_partitions_to_actors(actors,
                                                                          gang_scheduling)
        assigned_partition_refs = [self._get_multiple_partition_refs(part_ids)
                                   for part_ids in assigned_partitions]
        new_parts_refs = [func(actor, part_ref)
                          for actor, part_ids in zip(actors, assigned_partition_refs)
                          for part_ref in part_ids]
        part_id2ip = {}
        for actor_idx, parts in enumerate(assigned_partitions):
            ip = actor_ips[actor_idx]
            for part_id in parts:
                part_id2ip[part_id] = ip
        new_parts_ids = [idx for part_ids in assigned_partitions for idx in part_ids]
        return RayRdd.from_partition_refs(new_parts_refs, new_parts_ids, part_id2ip)

    def zip_partitions_with_actors(self, ray_rdd, actors, func, gang_scheduling=True):
        assert self.num_partitions() == ray_rdd.num_partitions(),\
            "the rdds to be zipped must have the same number of partitions"
        assigned_partitions, actor_ips = self.assign_partitions_to_actors(actors,
                                                                          gang_scheduling)
        assigned_partition_refs = [self._get_multiple_partition_refs(part_ids)
                                   for part_ids in assigned_partitions]
        assigned_partition_refs_other = [ray_rdd._get_multiple_partition_refs(part_ids)
                                         for part_ids in assigned_partitions]

        partitions_refs_tuple = zip(assigned_partition_refs, assigned_partition_refs_other)
        new_parts_refs = [func(actor, this_part_ref, that_part_ref)
                          for actor, (this_part_refs, that_part_refs) in zip(actors,
                                                                             partitions_refs_tuple)
                          for this_part_ref, that_part_ref in zip(this_part_refs, that_part_refs)]
        part_id2ip = {}
        for actor_idx, parts in enumerate(assigned_partitions):
            ip = actor_ips[actor_idx]
            for part_id in parts:
                part_id2ip[part_id] = ip
        new_parts_ids = [idx for part_ids in assigned_partitions for idx in part_ids]
        return RayRdd.from_partition_refs(new_parts_refs, new_parts_ids, part_id2ip)

    def assign_partitions_to_actors(self, actors, one_to_one=True):
        num_parts = ray.get(self.meta_store.num_partitions.remote())
        if num_parts < len(actors):
            raise ValueError(f"this rdd has {num_parts} partitions, which is smaller"
                             f"than actor number ({len(actors)} actors).")

        avg_part_num = num_parts // len(actors)
        remainder = num_parts % len(actors)

        if one_to_one:
            assert avg_part_num == 1 and remainder == 0,\
                "there must be the same number of actors and partitions," \
                f" got actor number: {len(actors)}, partition number: {num_parts}"

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
            assert hasattr(actor, "get_node_ip"), "each actor should have a get_node_ip method"
            actor_ip = actor.get_node_ip.remote()
            actor_ips.append(actor_ip)

        actor_ips = ray.get(actor_ips)

        actor2assignments = [[] for i in range(len(actors))]

        ip2actors = {}
        for idx, ip in enumerate(actor_ips):
            if ip not in ip2actors:
                ip2actors[ip] = []
            ip2actors[ip].append(idx)

        unassinged = []
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
                unassinged.append((part_idx, ip))

        for part_idx, ip in unassinged:
            for current_assignments in actor2assignments:
                if len(current_assignments) < avg_part_num:
                    current_assignments.append(part_idx)
                elif len(current_assignments) == avg_part_num and remainder > 0:
                    current_assignments.append(part_idx)
                    remainder -= 1
        return actor2assignments, actor_ips

    @staticmethod
    def from_partition_refs(parts_refs, part_ids, part_id2ip):
        ray_ctx = RayContext.get()
        uuid_str = str(uuid.uuid4())
        meta_store = MetaStore.options(name=f"meta_store:{uuid_str}").remote()

        results = []
        for part_id, part_ref in zip(part_ids, parts_refs):
            result = meta_store.set_partition_ref.remote(part_id, [part_ref])
            results.append(result)
        ray.get(results)

        return RayRdd(uuid_str, meta_store, part_id2ip)

    @staticmethod
    def from_spark_rdd(rdd):
        return RayRdd._from_spark_rdd_ray_api(rdd)

    @staticmethod
    def _from_spark_rdd_ray_api(rdd):
        ray_ctx = RayContext.get()
        address = ray_ctx.redis_address
        password = ray_ctx.redis_password
        driver_ip = ray.services.get_node_ip_address()
        uuid_str = str(uuid.uuid4())
        meta_store = MetaStore.options(name=f"meta_store:{uuid_str}").remote()
        resources = ray.cluster_resources()
        nodes = []
        for key, value in resources.items():
            if key.startswith("node:"):
                # if running in cluster, filter out driver ip
                if not (not ray_ctx.is_local and key == f"node:{driver_ip}"):
                    nodes.append(key)

        partition_stores = {}
        for node in nodes:
            name = f"partition:{uuid_str}:{node}"
            store = ray.remote(num_cpus=0, resources={node: 1e-4})(PartitionUploader)\
                .options(name=name).remote(meta_store)
            partition_stores[name] = store
        partition_store_names = list(partition_stores.keys())
        id2ip = rdd.mapPartitionsWithIndex(lambda idx, part: write_to_ray(
            idx, part, address, password, partition_store_names)).collect()

        return RayRdd(uuid_str, meta_store, dict(id2ip))
