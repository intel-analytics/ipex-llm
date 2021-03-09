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
import pytest
import ray

from zoo.orca.data.ray_xshards import RayXShards


def get_ray_xshards():
    from zoo.orca.data import XShards
    import numpy as np

    ndarray_dict = {"x": np.random.randn(10, 4), "y": np.random.randn(10, 4)}

    spark_xshards = XShards.partition(ndarray_dict)

    ray_xshards = RayXShards.from_spark_xshards(spark_xshards)

    return ray_xshards, ndarray_dict


def verify_collect_results(data_parts, ndarray_dict):
    import numpy as np
    for k, array in ndarray_dict.items():
        reconstructed = np.concatenate([part[k] for part in data_parts])
        assert np.allclose(array, reconstructed)


def test_from_spark_xshards(orca_context_fixture):
    ray_xshards, ndarray_dict = get_ray_xshards()
    data_parts = ray_xshards.collect()
    verify_collect_results(data_parts, ndarray_dict)


def test_to_spark_xshards(orca_context_fixture):
    ray_xshards, ndarray_dict = get_ray_xshards()
    data_parts = ray_xshards.to_spark_xshards().collect()
    verify_collect_results(data_parts, ndarray_dict)


@ray.remote
class Add1Actor:

    def get_node_ip(self):
        import ray
        return ray._private.services.get_node_ip_address()

    def add_one(self, partition):
        return [{k: (value + 1) for k, value in shards.items()} for shards in partition]


def test_assign_partitions_to_actors(orca_context_fixture):
    ray_xshards, _ = get_ray_xshards()
    part_num = ray_xshards.num_partitions()

    actor_num = 3
    actors = [Add1Actor.remote() for i in range(actor_num)]
    parts_list, _ = ray_xshards.assign_partitions_to_actors(actors, one_to_one=False)

    assert len(parts_list) == actor_num

    div, mod = divmod(part_num, actor_num)
    for counter in range(actor_num):
        if counter < mod:
            assert len(parts_list[counter]) == div + 1
        else:
            assert len(parts_list[counter]) == div


def test_assign_partitions_to_actors_one_to_one_fail(orca_context_fixture):
    ray_xshards, _ = get_ray_xshards()
    part_num = ray_xshards.num_partitions()

    actors = [Add1Actor.remote() for i in range(part_num - 1)]
    with pytest.raises(AssertionError) as excinfo:
        parts_list, _ = ray_xshards.assign_partitions_to_actors(actors, one_to_one=True)

        assert excinfo.match("there must be the same number of actors and partitions")


def test_transform_shards_with_actors(orca_context_fixture):
    ray_xshards, ndarray_dict = get_ray_xshards()
    ndarray_dict_mapped = {k: value + 1 for k, value in ndarray_dict.items()}

    actors = [Add1Actor.remote() for i in range(3)]
    map_func = lambda actor, part_ref: actor.add_one.remote(part_ref)
    result_xshards = ray_xshards.transform_shards_with_actors(actors, map_func,
                                                              gang_scheduling=False)
    results = result_xshards.collect()
    verify_collect_results(results, ndarray_dict_mapped)


def test_transform_shards_with_actors_gang_scheduling_fail(orca_context_fixture):
    ray_xshards, ndarray_dict = get_ray_xshards()
    part_num = ray_xshards.num_partitions()

    actors = [Add1Actor.remote() for i in range(part_num - 1)]
    map_func = lambda actor, part_ref: actor.add_one.remote(part_ref)
    with pytest.raises(AssertionError) as excinfo:
        ray_xshards.transform_shards_with_actors(actors, map_func)
        assert excinfo.match("there must be the same number of actors and partitions")


def test_transform_shards_with_actors_gang_scheduling(orca_context_fixture):
    ray_xshards, ndarray_dict = get_ray_xshards()
    part_num = ray_xshards.num_partitions()
    ndarray_dict_mapped = {k: value + 1 for k, value in ndarray_dict.items()}

    actors = [Add1Actor.remote() for i in range(part_num)]
    map_func = lambda actor, part_ref: actor.add_one.remote(part_ref)
    result_xshards = ray_xshards.transform_shards_with_actors(actors, map_func,
                                                              gang_scheduling=True)
    results = result_xshards.collect()

    verify_collect_results(results, ndarray_dict_mapped)


if __name__ == "__main__":
    pytest.main([__file__])
