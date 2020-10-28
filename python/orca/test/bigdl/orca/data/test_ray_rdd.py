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

from zoo.orca.data.ray_rdd import RayRdd


def test_from_spark_rdd(orca_context_fixture):
    sc = orca_context_fixture
    rdd = sc.parallelize(range(1000))

    ray_rdd = RayRdd.from_spark_rdd(rdd)

    results = ray_rdd.collect()

    assert results == list(range(1000))


def test_to_spark_rdd(orca_context_fixture):
    sc = orca_context_fixture
    rdd = sc.parallelize(range(1000))

    ray_rdd = RayRdd.from_spark_rdd(rdd)

    results = ray_rdd.to_spark_rdd().collect()

    assert results == list(range(1000))


@ray.remote
class Add1Actor:

    def get_node_ip(self):
        import ray
        return ray.services.get_node_ip_address()

    def add_one(self, partition):
        return [i + 1 for i in partition]


def test_assign_partitions_to_actors(orca_context_fixture):

    sc = orca_context_fixture
    rdd = sc.parallelize(range(1000), 7)

    ray_rdd = RayRdd.from_spark_rdd(rdd)

    actors = [Add1Actor.remote() for i in range(3)]
    parts_list, _ = ray_rdd.assign_partitions_to_actors(actors, one_to_one=False)

    print(parts_list)

    assert len(parts_list) == 3
    assert len(parts_list[0]) == 3
    assert len(parts_list[1]) == 2
    assert len(parts_list[2]) == 2


def test_assign_partitions_to_actors_one_to_one_fail(orca_context_fixture):

    sc = orca_context_fixture
    rdd = sc.parallelize(range(1000), 7)

    ray_rdd = RayRdd.from_spark_rdd(rdd)

    actors = [Add1Actor.remote() for i in range(3)]
    with pytest.raises(AssertionError) as excinfo:
        parts_list, _ = ray_rdd.assign_partitions_to_actors(actors, one_to_one=True)

        assert excinfo.match("there must be the same number of actors and partitions")


def test_map_partitions_with_actors(orca_context_fixture):

    sc = orca_context_fixture
    rdd = sc.parallelize(range(1000), 7)

    ray_rdd = RayRdd.from_spark_rdd(rdd)

    actors = [Add1Actor.remote() for i in range(3)]
    map_func = lambda actor, part_ref: actor.add_one.remote(part_ref)
    result_rdd = ray_rdd.map_partitions_with_actors(actors, map_func, gang_scheduling=False)
    results = result_rdd.collect()

    assert results == list(range(1, 1001))


def test_map_partitions_with_actors_gang_scheduling_fail(orca_context_fixture):

    sc = orca_context_fixture
    rdd = sc.parallelize(range(1000), 7)

    ray_rdd = RayRdd.from_spark_rdd(rdd)

    actors = [Add1Actor.remote() for i in range(3)]
    map_func = lambda actor, part_ref: actor.add_one.remote(part_ref)
    with pytest.raises(AssertionError) as excinfo:
        result_rdd = ray_rdd.map_partitions_with_actors(actors, map_func)
        assert excinfo.match("there must be the same number of actors and partitions")


def test_map_partitions_with_actors_gang_scheduling(orca_context_fixture):

    sc = orca_context_fixture
    rdd = sc.parallelize(range(1000), 7)

    ray_rdd = RayRdd.from_spark_rdd(rdd)

    actors = [Add1Actor.remote() for i in range(7)]
    map_func = lambda actor, part_ref: actor.add_one.remote(part_ref)
    result_rdd = ray_rdd.map_partitions_with_actors(actors, map_func, gang_scheduling=True)
    results = result_rdd.collect()

    assert results == list(range(1, 1001))


if __name__ == "__main__":
    pytest.main([__file__])
