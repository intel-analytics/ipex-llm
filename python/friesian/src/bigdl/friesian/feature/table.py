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
import os

from pyspark.sql.types import IntegerType, ShortType, LongType, FloatType, DecimalType, \
    DoubleType, ArrayType, DataType
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col as pyspark_col, udf, array, broadcast, lit
from pyspark.sql import Row
import pyspark.sql.functions as F

from zoo.orca import OrcaContext
from zoo.friesian.feature.utils import *
from zoo.common.utils import callZooFunc


JAVA_INT_MIN = -2147483648
JAVA_INT_MAX = 2147483647


class Table:
    def __init__(self, df):
        self.df = df
        self.__column_names = self.df.schema.names

    @staticmethod
    def _read_parquet(paths):
        if not isinstance(paths, list):
            paths = [paths]
        spark = OrcaContext.get_spark_session()
        df = spark.read.parquet(*paths)
        return df

    @staticmethod
    def _read_json(paths, cols):
        if not isinstance(paths, list):
            paths = [paths]
        spark = OrcaContext.get_spark_session()
        df = spark.read.json(paths)
        if cols:
            if isinstance(cols, list):
                df = df.select(*cols)
            elif isinstance(cols, str):
                df = df.select(cols)
            else:
                raise Exception("cols should be a column name or list of column names")
        return df

    def _clone(self, df):
        return Table(df)

    def compute(self):
        """
        Trigger computation of the Table.
        """
        compute(self.df)
        return self

    def to_spark_df(self):
        """
        Convert the current Table to a Spark DataFrame.

        :return: The converted Spark DataFrame.
        """
        return self.df

    def size(self):
        """
        Returns the number of rows in this Table.

        :return: The number of rows in the current Table.
        """
        cnt = self.df.count()
        return cnt

    def broadcast(self):
        """
        Marks the Table as small enough for use in broadcast join.
        """
        self.df = broadcast(self.df)

    def select(self, *cols):
        """
        Select specific columns.

        :param cols: a string or a list of strings that specifies column names. If it is '*',
                     select all the columns.

        :return: A new Table that contains the specified columns.
        """
        # If cols is None, it makes more sense to raise error
        # instead of returning an empty Table.
        if not cols:
            raise ValueError("cols should be str or a list of str, but got None.")
        return self._clone(self.df.select(*cols))

    def drop(self, *cols):
        """
        Returns a new Table that drops the specified column.
        This is a no-op if schema doesn't contain the given column name(s).

        :param cols: a string name of the column to drop, or a list of string name of the columns
               to drop.

        :return: A new Table that drops the specified column.
        """
        return self._clone(self.df.drop(*cols))

    def fillna(self, value, columns):
        """
        Replace null values.

        :param value: int, long, float, string, or boolean.
               Value to replace null values with.
        :param columns: list of str, the target columns to be filled. If columns=None and value
               is int, all columns of integer type will be filled. If columns=None and value is
               long, float, string or boolean, all columns will be filled.

        :return: A new Table that replaced the null values with specified value
        """
        if columns and not isinstance(columns, list):
            columns = [columns]
        if columns:
            check_col_exists(self.df, columns)
        if isinstance(value, int) and JAVA_INT_MIN <= value <= JAVA_INT_MAX:
            if columns:
                col_not_int_list = list(filter(lambda x: x[0] in columns and x[1] != "int",
                                               self.df.dtypes))
                if len(col_not_int_list) == 0:
                    return self._clone(fill_na_int(self.df, value, columns))
            else:
                return self._clone(fill_na_int(self.df, value, columns))
        return self._clone(fill_na(self.df, value, columns))

    def dropna(self, columns, how='any', thresh=None):
        """
        Drops the rows containing null values in the specified columns.

        :param columns: a string or a list of strings that specifies column names. If it is None,
               it will operate on all columns.
        :param how: If `how` is "any", then drop rows containing any null values in `columns`.
               If `how` is "all", then drop rows only if every column in `columns` is null for
               that row.
        :param thresh: int, if specified, drop rows that have less than thresh non-null values.
               Default is None.

        :return: A new Table that drops the rows containing null values in the specified columns.
        """
        return self._clone(self.df.dropna(how, thresh, subset=columns))

    def distinct(self):
        """
        Select the distinct rows of the Table.

        :return: A new Table that only contains distinct rows.
        """
        return self._clone(self.df.distinct())

    def filter(self, condition):
        """
        Filters the rows that satisfy `condition`. For instance, filter("col_1 == 1") will filter
        the rows that has value 1 at column col_1.

        :param condition: a string that gives the condition for filtering.

        :return: A new Table with filtered rows.
        """
        return self._clone(self.df.filter(condition))

    def clip(self, columns, min=None, max=None):
        """
        Clips continuous values so that they are within the range [min, max]. For instance, by
        setting the min value to 0, all negative values in columns will be replaced with 0.

        :param columns: str or list of str, the target columns to be clipped.
        :param min: numeric, the minimum value to clip values to. Values less than this will be
               replaced with this value.
        :param max: numeric, the maximum value to clip values to. Values greater than this will be
               replaced with this value.

        :return: A new Table that replaced the value less than `min` with specified `min` and the
                 value greater than `max` with specified `max`.
        """
        assert min is not None or max is not None, "at least one of min and max should be not None"
        if columns is None:
            raise ValueError("columns should be str or list of str, but got None.")
        if not isinstance(columns, list):
            columns = [columns]
        check_col_exists(self.df, columns)
        return self._clone(clip(self.df, columns, min, max))

    def log(self, columns, clipping=True):
        """
        Calculates the log of continuous columns.

        :param columns: str or list of str, the target columns to calculate log.
        :param clipping: boolean, if clipping=True, the negative values in columns will be
               clipped to 0 and `log(x+1)` will be calculated. If clipping=False, `log(x)` will be
               calculated.

        :return: A new Table that replaced value in columns with logged value.
        """
        if columns is None:
            raise ValueError("columns should be str or list of str, but got None.")
        if not isinstance(columns, list):
            columns = [columns]
        check_col_exists(self.df, columns)
        return self._clone(log_with_clip(self.df, columns, clipping))

    def fill_median(self, columns):
        """
        Replaces null values with the median in the specified numeric columns. Any column to be
        filled should not contain only null values.

        :param columns: a string or a list of strings that specifies column names. If it is None,
               it will operate on all numeric columns.

        :return: A new Table that replaces null values with the median in the specified numeric
                 columns.
        """
        if columns and not isinstance(columns, list):
            columns = [columns]
        if columns:
            check_col_exists(self.df, columns)
        return self._clone(fill_median(self.df, columns))

    def median(self, columns):
        """
        Returns a new Table that has two columns, `column` and `median`, containing the column
        names and the medians of the specified numeric columns.

        :param columns: a string or a list of strings that specifies column names. If it is None,
               it will operate on all numeric columns.

        :return: A new Table that contains the medians of the specified columns.
        """
        if columns and not isinstance(columns, list):
            columns = [columns]
        if columns:
            check_col_exists(self.df, columns)
        return self._clone(median(self.df, columns))

    # Merge column values as a list to a new col
    def merge_cols(self, columns, target):
        """
        Merge column values as a list to a new col.

        :param columns: list of str, the target columns to be merged.
        :param target: str, the new column name of the merged column.

        :return: A new Table that replaced columns with a new target column of merged list value.
        """
        assert isinstance(columns, list)
        return self._clone(self.df.withColumn(target, array(columns)).drop(*columns))

    def rename(self, columns):
        """
        Rename columns with new column names

        :param columns: dict. Name pairs. For instance, {'old_name1': 'new_name1', 'old_name2':
               'new_name2'}".

        :return: A new Table with new column names.
        """
        assert isinstance(columns, dict), "columns should be a dictionary of {'old_name1': " \
                                          "'new_name1', 'old_name2': 'new_name2'}"
        new_df = self.df
        for old_name, new_name in columns.items():
            new_df = new_df.withColumnRenamed(old_name, new_name)
        return self._clone(new_df)

    def show(self, n=20, truncate=True):
        """
        Prints the first `n` rows to the console.

        :param n: int, the number of rows to show.
        :param truncate: If set to True, truncate strings longer than 20 chars by default.
               If set to a number greater than one, truncates long strings to length `truncate` and
               align cells right.
        """
        self.df.show(n, truncate)

    def add(self, columns, value=1):
        """
        Increase all of values of the target numeric column(s) by a constant value.

        :param columns: str or list of str, the target columns to be increased.
        :param value: numeric (int/float/double/short/long), the constant value to be added.

        :return: A new Table with updated numeric values on specified columns.
        """
        if columns is None:
            raise ValueError("Columns should be str or list of str, but got None")
        if not isinstance(columns, list):
            columns = [columns]
        check_col_exists(self.df, columns)
        new_df = self.df
        for column in columns:
            if new_df.schema[column].dataType not in [IntegerType(), ShortType(),
                                                      LongType(), FloatType(),
                                                      DecimalType(), DoubleType()]:
                raise ValueError("Column type should be numeric, but have type {} \
                    for column {}".format(new_df.schema[column].dataType, column))
            new_df = new_df.withColumn(column, pyspark_col(column) + lit(value))
        return self._clone(new_df)

    @property
    def columns(self):
        """
        Get column names of the Table.

        :return: A list of strings that specify column names.
        """
        return self.__column_names

    def sample(self, fraction, replace=False, seed=None):
        """
        Return a sampled subset of Table.

        :param fraction: float, fraction of rows to generate, should be within the
        range [0, 1].
        :param replace: allow or disallow sampling of the same row more than once.
        :param seed: seed for sampling.

        :return: A new Table with sampled rows.
        """
        return self._clone(self.df.sample(withReplacement=replace, fraction=fraction, seed=seed))

    def write_parquet(self, path, mode="overwrite"):
        self.df.write.mode(mode).parquet(path)

    def cast(self, columns, type):
        """
        Cast columns to the specified type.

        :param columns: a string or a list of strings that specifies column names.
                        If it is None, then cast all of the columns.
        :param type: a string ("string", "boolean", "int", "long", "short", "float", "double")
                     that specifies the type.

        :return: A new Table that casts all of the specified columns to the specified type.
        """
        if columns is None:
            columns = self.df.columns
        elif not isinstance(columns, list):
            columns = [columns]
            check_col_exists(self.df, columns)
        valid_types = ["str", "string", "bool", "boolean", "int",
                       "integer", "long", "short", "float", "double"]
        if not (isinstance(type, str) and (type in valid_types)) \
           and not isinstance(type, DataType):
            raise ValueError(
                "type should be string, boolean, int, long, short, float, double.")
        transform_dict = {"str": "string", "bool": "boolean", "integer": "int"}
        type = transform_dict[type] if type in transform_dict else type
        df_cast = self._clone(self.df)
        for i in columns:
            df_cast.df = df_cast.df.withColumn(i, pyspark_col(i).cast(type))
        return df_cast

    def __getattr__(self, name):
        return self.df.__getattr__(name)

    def col(self, name):
        return pyspark_col(name)


class FeatureTable(Table):
    @classmethod
    def read_parquet(cls, paths):
        """
        Loads Parquet files as a FeatureTable.

        :param paths: str or a list of str. The path/paths to Parquet file(s).

        :return: A FeatureTable for recommendation data.
        """
        return cls(Table._read_parquet(paths))

    @classmethod
    def read_json(cls, paths, cols=None):
        return cls(Table._read_json(paths, cols))

    def encode_string(self, columns, indices):
        """
        Encode columns with provided list of StringIndex.

        :param columns: str or a list of str, target columns to be encoded.
        :param indices: StringIndex or a list of StringIndex, StringIndexes of target columns.
               The StringIndex should at least have two columns: id and the corresponding
               categorical column.
               Or it can be a dict or a list of dicts. In this case,
               the keys of the dict should be within the categorical column
               and the values are the target ids to be encoded.

        :return: A new FeatureTable which transforms categorical features into unique integer
                 values with provided StringIndexes.
        """
        if not isinstance(columns, list):
            columns = [columns]
        if not isinstance(indices, list):
            indices = [indices]
        assert len(columns) == len(indices)
        if isinstance(indices[0], dict):
            indices = list(map(lambda x: StringIndex.from_dict(x[1], columns[x[0]]),
                               enumerate(indices)))
        data_df = self.df
        for i in range(len(columns)):
            index_tbl = indices[i]
            col_name = columns[i]
            index_tbl.broadcast()
            data_df = data_df.join(index_tbl.df, col_name, how="left") \
                .drop(col_name).withColumnRenamed("id", col_name)\
                .dropna(subset=[col_name])
        return FeatureTable(data_df)

    def gen_string_idx(self, columns, freq_limit):
        """
        Generate unique index value of categorical features.

        :param columns: str or a list of str, target columns to generate StringIndex.
        :param freq_limit: int, dict or None. Categories with a count/frequency below freq_limit
               will be omitted from the encoding. Can be represented as both an integer,
               dict or None. For instance, 15, {'col_4': 10, 'col_5': 2} etc. None means all the
               categories that appear will be encoded.

        :return: A list of StringIndex.
        """
        if columns is None:
            raise ValueError("columns should be str or list of str, but got None.")
        if not isinstance(columns, list):
            columns = [columns]
        check_col_exists(self.df, columns)
        if freq_limit:
            if isinstance(freq_limit, int):
                freq_limit = str(freq_limit)
            elif isinstance(freq_limit, dict):
                freq_limit = ",".join(str(k) + ":" + str(v) for k, v in freq_limit.items())
            else:
                raise ValueError("freq_limit only supports int, dict or None, but get " +
                                 freq_limit.__class__.__name__)
        df_id_list = generate_string_idx(self.df, columns, freq_limit)
        string_idx_list = list(map(lambda x: StringIndex(x[0], x[1]),
                                   zip(df_id_list, columns)))
        return string_idx_list

    def gen_ind2ind(self, cols, indices):
        """
        Generate a mapping between of indices

        :param cols: a list of str, target columns to generate StringIndex.
        :param indices:  list of StringIndex

        :return: FeatureTable
        """
        df = self.encode_string(cols, indices).df.select(*cols).distinct()
        return FeatureTable(df)

    def _clone(self, df):
        return FeatureTable(df)

    def cross_columns(self, crossed_columns, bucket_sizes):
        """
        Cross columns and hashed to specified bucket size
        :param crossed_columns: list of column name pairs to be crossed.
        i.e. [['a', 'b'], ['c', 'd']]
        :param bucket_sizes: hash bucket size for crossed pairs. i.e. [1000, 300]
        :return: FeatureTable include crossed columns(i.e. 'a_b', 'c_d')
        """
        df = cross_columns(self.df, crossed_columns, bucket_sizes)
        return FeatureTable(df)

    def normalize(self, columns):
        """
        Normalize numeric columns
        :param columns: list of column names
        :return: FeatureTable
        """
        df = self.df
        types = [x[1] for x in self.df.select(*columns).dtypes]
        scalar_cols = [columns[i] for i in range(len(columns))
                       if types[i] == "int" or types[i] == "bigint"
                       or types[i] == "float" or types[i] == "double"]
        array_cols = [columns[i] for i in range(len(columns))
                      if types[i] == "array<int>" or types[i] == "array<bigint>"
                      or types[i] == "array<float>" or types[i] == "array<double>"]
        vector_cols = [columns[i] for i in range(len(columns)) if types[i] == "vector"]
        if scalar_cols:
            assembler = VectorAssembler(inputCols=scalar_cols, outputCol="vect")

            # MinMaxScaler Transformation
            scaler = MinMaxScaler(inputCol="vect", outputCol="scaled")

            # Pipeline of VectorAssembler and MinMaxScaler
            pipeline = Pipeline(stages=[assembler, scaler])

            tolist = udf(lambda x: x.toArray().tolist(), ArrayType(DoubleType()))

            # Fitting pipeline on dataframe
            df = pipeline.fit(df).transform(df) \
                .withColumn("scaled_list", tolist(pyspark_col("scaled"))) \
                .drop("vect").drop("scaled")
            for i in range(len(scalar_cols)):
                df = df.withColumn(scalar_cols[i], pyspark_col("scaled_list")[i])
            df = df.drop("scaled_list")

            # cast to float
            for c in scalar_cols:
                df = df.withColumn(c, pyspark_col(c).cast("float"))

        for c in array_cols:
            df = normalize_array(df, c)

        for c in vector_cols:
            scaler = MinMaxScaler(inputCol=c, outputCol="scaled")
            df = scaler.fit(df).transform(df).withColumnRenamed("scaled", c)

        return FeatureTable(df)

    def add_negative_samples(self, item_size, item_col="item", label_col="label", neg_num=1):
        """
        Generate negative item visits for each positive item visit

        :param item_size: integer, max of item.
        :param item_col:  string, name of item column
        :param label_col:  string, name of label column
        :param neg_num:  integer, for each positive record, add neg_num of negative samples

        :return: FeatureTable
        """
        df = callZooFunc("float", "addNegSamples", self.df, item_size, item_col, label_col, neg_num)
        return FeatureTable(df)

    def add_hist_seq(self, user_col, cols, sort_col='time', min_len=1, max_len=100):
        """
        Generate a list of item visits in history

        :param user_col: string, user column.
        :param cols:  list of string, ctolumns need to be aggragated
        :param sort_col:  string, sort by sort_col
        :param min_len:  int, minimal length of a history list
        :param max_len:  int, maximal length of a history list

        :return: FeatureTable
        """
        df = callZooFunc("float", "addHistSeq", self.df, user_col, cols, sort_col, min_len, max_len)
        return FeatureTable(df)

    def add_neg_hist_seq(self, item_size, item_history_col, neg_num):
        """
        Generate a list negative samples for each item in item_history_col

        :param item_size: int, max of item.
        :param item2cat:  FeatureTable with a dataframe of item to catgory mapping
        :param item_history_col:  string, this column should be a list of visits in history
        :param neg_num:  int, for each positive record, add neg_num of negative samples

        :return: FeatureTable
        """
        df = callZooFunc("float", "addNegHisSeq", self.df, item_size, item_history_col, neg_num)
        return FeatureTable(df)

    def pad(self, padding_cols, seq_len=100):
        """
        Post padding padding columns

        :param padding_cols: list of string, columns need to be padded with 0s.
        :param seq_len:  int, length of padded column

        :return: FeatureTable
        """
        df = callZooFunc("float", "postPad", self.df, padding_cols, seq_len)
        return FeatureTable(df)

    def mask(self, mask_cols, seq_len=100):
        """
        Mask mask_cols columns

        :param mask_cols: list of string, columns need to be masked with 1s and 0s.
        :param seq_len:  int, length of masked column

        :return: FeatureTable
        """
        df = callZooFunc("float", "mask", self.df, mask_cols, seq_len)
        return FeatureTable(df)

    def add_length(self, col_name):
        """
        Generate the length of a columb.

        :param col_name: string.

        :return: FeatureTable
        """
        df = callZooFunc("float", "addLength", self.df, col_name)
        return FeatureTable(df)

    def mask_pad(self, padding_cols, mask_cols, seq_len=100):
        """
        Mask and pad columns

        :param padding_cols: list of string, columns need to be padded with 0s.
        :param mask_cols: list of string, columns need to be masked with 1s and 0s.
        :param seq_len:  int, length of masked column

        :return: FeatureTable
        """
        table = self.mask(mask_cols, seq_len)
        return table.pad(padding_cols, seq_len)

    def transform_python_udf(self, in_col, out_col, udf_func):
        """
        Transform a FeatureTable using a python udf

        :param in_col: string, name of column needed to be transformed.
        :param out_col: string, output column.
        :param udf_func: user defined python function

        :return: FeatureTable
        """
        df = self.df.withColumn(out_col, udf_func(pyspark_col(in_col)))
        return FeatureTable(df)

    def join(self, table, on=None, how=None):
        """
        Join a FeatureTable with another FeatureTable, it is wrapper of spark dataframe join

        :param table: FeatureTable
        :param on: string, join on this column
        :param how: string

        :return: FeatureTable
        """
        assert isinstance(table, Table), "the joined table should be a Table"
        joined_df = self.df.join(table.df, on=on, how=how)
        return FeatureTable(joined_df)

    def add_feature(self, item_cols, feature_tbl, default_value):
        """
        Get the category or other field from another map like FeatureTable

        :param item_cols: list[string]
        :param feature_tbl: FeatureTable with two columns [category, item]
        :param defalut_cat_index: default value for category if key does not exist

        :return: FeatureTable
        """
        item2cat_map = dict(feature_tbl.df.distinct().rdd.map(lambda row: (row[0], row[1]))
                            .collect())

        def gen_cat(items):
            getcat = lambda item: item2cat_map.get(item, default_value)
            if isinstance(items, int):
                cats = getcat(items)
            elif isinstance(items, list) and isinstance(items[0], int):
                cats = [getcat(item) for item in items]
            elif isinstance(items, list) and isinstance(items[0], list) and isinstance(items[0][0],
                                                                                       int):
                cats = []
                for line in items:
                    line_cats = [getcat(item) for item in line]
                    cats.append(line_cats)
            else:
                raise ValueError('only int, list[int], and list[list[int]] are supported.')
            return cats

        df = self.df
        for c in item_cols:
            col_type = df.schema[c].dataType
            cat_udf = udf(gen_cat, col_type)
            df = df.withColumn(c.replace("item", "category"), cat_udf(pyspark_col(c)))
        return FeatureTable(df)

    def group_by(self, columns=[], agg="count", join=False):
        """
        Group the Table with specified columns and then run aggregation. Optionally join the result
        with the original Table.

        :param columns: str or list of str. Columns to group the Table. If it is an empty list,
               aggregation is run directly without grouping. Default is [].
        :param agg: str, list or dict. Aggragate functions to be applied to grouped Table.
               Default is "count".
               Supported aggregate functions are: "max", "min", "count", "sum", "avg", "mean",
               "sumDistinct", "stddev", "stddev_pop", "variance", "var_pop", "skewness", "kurtosis",
               "collect_list", "collect_set", "approx_count_distinct", "first", "last".
               If agg is a str, then agg is the aggregate function and the aggregation is performed
               on all columns that are not in `columns`.
               If agg is a list of str, then agg is a list of aggregate function and the aggregation
               is performed on all columns that are not in `columns`.
               If agg is a single dict mapping from string to string, then the key is the column
               to perform aggregation on, and the value is the aggregate function.
               If agg is a single dict mapping from string to list, then the key is the
               column to perform aggregation on, and the value is list of aggregate functions.

               Examples:
               agg="sum"
               agg=["last", "stddev"]
               agg={"*":"count"}
               agg={"col_1":"sum", "col_2":["count", "mean"]}
        :param join: boolean. If join is True, join the aggragation result with original Table.

        :return: A new Table with aggregated column fields.
        """
        if isinstance(columns, str):
            columns = [columns]
        assert isinstance(columns, list), "columns should be str or list of str"
        grouped_data = self.df.groupBy(columns)

        if isinstance(agg, str):
            agg_exprs_dict = {agg_column: agg for agg_column in self.df.columns
                              if agg_column not in columns}
            agg_df = grouped_data.agg(agg_exprs_dict)
        elif isinstance(agg, list):
            agg_exprs_list = []
            for stat in agg:
                stat_func = getattr(F, stat)
                agg_exprs_list += [stat_func(agg_column) for agg_column in self.df.columns
                                   if agg_column not in columns]
            agg_df = grouped_data.agg(*agg_exprs_list)
        elif isinstance(agg, dict):
            if all(isinstance(stats, str) for agg_column, stats in agg.items()):
                agg_df = grouped_data.agg(agg)
            else:
                agg_exprs_list = []
                for agg_column, stats in agg.items():
                    if isinstance(stats, str):
                        stats = [stats]
                    assert isinstance(stats, list), "value in agg should be str or list of str"
                    for stat in stats:
                        stat_func = getattr(F, stat)
                        agg_exprs_list += [stat_func(agg_column)]
                agg_df = grouped_data.agg(*agg_exprs_list)
        else:
            raise TypeError("agg should be str, list of str, or dict")

        if join:
            assert columns, "columns can not be empty if join is True"
            result_df = self.df.join(agg_df, on=columns, how="left")
            return FeatureTable(result_df)
        else:
            return FeatureTable(agg_df)


class StringIndex(Table):
    def __init__(self, df, col_name):
        super().__init__(df)
        cols = df.columns
        assert len(cols) >= 2, "StringIndex should have >= 2 columns: col_name, id and other " \
                               "columns"
        assert "id" in cols, "id should be a column of the DataFrame"
        assert col_name in cols, col_name + " should be a column of the DataFrame"
        self.col_name = col_name

    @classmethod
    def read_parquet(cls, paths, col_name=None):
        """
        Loads Parquet files as a StringIndex.

        :param paths: str or a list of str. The path/paths to Parquet file(s).
        :param col_name: str. The column name of the corresponding categorical column. If
               col_name is None, the file name will be used as col_name.

        :return: A StringIndex.
        """
        if not isinstance(paths, list):
            paths = [paths]
        if col_name is None and len(paths) >= 1:
            col_name = os.path.basename(paths[0]).split(".")[0]
        return cls(Table._read_parquet(paths), col_name)

    @classmethod
    def from_dict(cls, indices, col_name):
        """
        Create the StringIndex from a dict of indices.

        :param indices: dict. The key is the categorical column,
                        the value is the corresponding index.
                        We assume that the key is a str and the value is a int.
        :param col_name: str. The column name of the categorical column.

        :return: A StringIndex.
        """
        spark = OrcaContext.get_spark_session()
        if not isinstance(indices, dict):
            raise ValueError('indices should be dict, but get ' + indices.__class__.__name__)
        if not col_name:
            raise ValueError('col_name should be str, but get None')
        if not isinstance(col_name, str):
            raise ValueError('col_name should be str, but get ' + col_name.__class__.__name__)
        indices = map(lambda x: {col_name: x[0], 'id': x[1]}, indices.items())
        df = spark.createDataFrame(Row(**x) for x in indices)
        return cls(df, col_name)

    def _clone(self, df):
        return StringIndex(df, self.col_name)

    def write_parquet(self, path, mode="overwrite"):
        """
        Write StringIndex to Parquet file.

        :param path: str. The path to the `folder` of the Parquet file. Note that the col_name
               will be used as basename of the Parquet file.
        :param mode: str. `append`, `overwrite`, `error` or `ignore`. `append`: Append contents
               of this StringIndex to existing data. `overwrite`: Overwrite existing data.
               `error`: Throw an exception if data already exists. `ignore`: Silently ignore this
               operation if data already exists.
        """
        path = path + "/" + self.col_name + ".parquet"
        self.df.write.parquet(path, mode=mode)
