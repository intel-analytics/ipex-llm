from bigdl.orca.data import SparkXShards
import tensorflow as tf


class TF2Dataset(object):
    def __init__(self, dataset):
        self.rdd = dataset.as_tf_dataset()

    def get_xshards(self):
        return SparkXShards(self.rdd)

    def get_ray_xshards(self, num_workers):
        from bigdl.orca.data.utils import process_spark_xshards
        xshards = self.get_xshards()
        return process_spark_xshards(xshards, num_workers)


class Dataset(object):

    """
    Represents a distributed set of elements backed by an RDD,
    which is created by applying tensorflow dataset transformations
    on each partitions.
    """

    def __init__(self, xshards, create_dataset_fn):
        self.xshards = xshards
        self.create_dataset_fn = create_dataset_fn

    def as_tf_dataset(self):
        create_dataset_fn = self.create_dataset_fn

        def to_dataset(iter):

            data_list = list(iter)
            if not data_list:
                return []

            from tensorflow.python.distribute.coordinator.values import serialize_dataset_to_graph
            datasets = [create_dataset_fn(data) for data in data_list]
            from functools import reduce
            dataset = reduce(lambda x, y: x.concatenate(y), datasets)
            ds_def = serialize_dataset_to_graph(dataset).numpy()
            elem_spec = dataset.element_spec
            return [{"ds_def": ds_def, "elem_spec": elem_spec}]

        tf_dataset_rdd = self.xshards.rdd.mapPartitions(to_dataset)
        return tf_dataset_rdd

    @staticmethod
    def from_tensor_slices(xshards):
        return TensorSliceDataset(xshards)

    @staticmethod
    def from_tensor_slices_with_tbl(tbl):
        from bigdl.friesian.feature import FeatureTable
        from bigdl.friesian.feature.utils import featuretable_to_xshards
        assert isinstance(tbl, FeatureTable), "only Friesian FeatureTable is supported"
        xshards = featuretable_to_xshards(tbl)
        return TensorSliceDataset(xshards)

    def map(self, map_func):

        return MapDataset(self, map_func)


class TensorSliceDataset(Dataset):

    def __init__(self, xshards):
        assert isinstance(xshards, SparkXShards), \
            "only datasets backed by a SparkXShards are supported"

        self.xshards = xshards

        def create_dataset_fn(data):
            return tf.data.Dataset.from_tensor_slices(data)
        super().__init__(xshards, create_dataset_fn)


class MapDataset(Dataset):

    def __init__(self, input_dataset, map_func):

        create_pre_dataset_fn = input_dataset.create_dataset_fn

        def create_dataset_fn(data):
            dataset = create_pre_dataset_fn(data)
            return dataset.map(map_func)
        super().__init__(xshards=input_dataset.xshards,
                         create_dataset_fn=create_dataset_fn)

