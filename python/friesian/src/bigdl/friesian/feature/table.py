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

from pyspark import SparkContext
from pyspark.sql.functions import col, udf, array, broadcast, explode, struct, collect_list
from zoo.orca import OrcaContext
from zoo.friesian.feature.utils import generate_string_idx, fill_na, \
    fill_na_int, compute, log_with_clip, clip_min
import random
from pyspark.sql.types import ArrayType, IntegerType, Row, StructType, StructField
from zoo.common.utils import callZooFunc

JAVA_INT_MIN = -2147483648
JAVA_INT_MAX = 2147483647


class Table:
    def __init__(self, df):
        self.df = df

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
        Trigger computation of Table.
        """
        compute(self.df)
        return self

    def to_spark_df(self):
        """
        Convert current Table to spark DataFrame

        :return: The converted spark DataFrame
        """
        return self.df

    def count(self):
        """
        Returns the number of rows in this Table.

        :return: The number of rows in current Table
        """
        cnt = self.df.count()
        return cnt

    def broadcast(self):
        """
        Marks a Table as small enough for use in broadcast joins
        """
        self.df = broadcast(self.df)

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
        if isinstance(value, int) and JAVA_INT_MIN <= value <= JAVA_INT_MAX:
            return self._clone(fill_na_int(self.df, value, columns))
        else:
            return self._clone(fill_na(self.df, value, columns))

    def dropna(self, how='any', thresh=None, subset=None):
        """
        Drop null values. a wrapper of dataframe dropna
        :return: A new Table that replaced the null values with specified value
        """
        return self._clone(self.df.dropna(how, thresh, subset))

    def distinct(self):
        """
        A wrapper of dataframe distinct
        :return: A new Table that only has distinct rows
        """
        return self._clone(self.df.distinct())

    def clip(self, columns, min=0):
        """
        clips continuous values so that they are within a min bound. For instance by setting the
        min value to 0, all negative values in columns will be replaced with 0.

        :param columns: list of str, the target columns to be clipped.
        :param min: int, The mininum value to clip values to: values less than this will be
               replaced with this value.

        :return: A new Table that replaced the value less than `min` with specified `min`
        """
        if not isinstance(columns, list):
            columns = [columns]
        return self._clone(clip_min(self.df, columns, min))

    def log(self, columns, clipping=True):
        """
        Calculates the log of continuous columns.

        :param columns: list of str, the target columns to calculate log.
        :param clipping: boolean, if clipping=True, the negative values in columns will be
               clipped to 0 and `log(x+1)` will be calculated. If clipping=False, `log(x)` will be
               calculated.

        :return: A new Table that replaced value in columns with logged value.
        """
        if not isinstance(columns, list):
            columns = [columns]
        return self._clone(log_with_clip(self.df, columns, clipping))

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
               'new_name2'}"

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

        :param n: int, number of rows to show.
        :param truncate: If set to True, truncate strings longer than 20 chars by default.
               If set to a number greater than one, truncates long strings to length `truncate` and
               align cells right.
        """
        self.df.show(n, truncate)

    def write_parquet(self, path, mode="overwrite"):
        self.df.write.mode(mode).parquet(path)


class FeatureTable(Table):
    @classmethod
    def read_parquet(cls, paths):
        """
        Loads Parquet files, returning the result as a `FeatureTable`.

        :param paths: str or a list of str. The path/paths to Parquet file(s).

        :return: A FeatureTable
        """
        return cls(Table._read_parquet(paths))

    @classmethod
    def read_json(cls, paths, cols=None):
        return cls(Table._read_json(paths, cols))

    def encode_string(self, columns, indices):
        """
        Encode columns with provided list of StringIndex

        :param columns: str or a list of str, target columns to be encoded.
        :param indices: StringIndex or a list of StringIndex, StringIndexes of target columns.
               The StringIndex should at least have two columns: id and the corresponding
               categorical column.

        :return: A new FeatureTable which transforms categorical features into unique integer
                 values with provided StringIndexes.
        """
        if not isinstance(columns, list):
            columns = [columns]
        if not isinstance(indices, list):
            indices = [indices]
        assert len(columns) == len(indices)
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
        Generate unique index value of categorical features

        :param columns: str or a list of str, target columns to generate StringIndex.
        :param freq_limit: int, dict or None. Categories with a count/frequency below freq_limit
               will be omitted from the encoding. Can be represented as both an integer,
               dict or None. For instance, 15, {'col_4': 10, 'col_5': 2} etc. None means all the
               categories that appear will be encoded.

        :return: List of StringIndex
        """
        if not isinstance(columns, list):
            columns = [columns]
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
         Generagte length of a colum

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
        df = self.df.withColumn(out_col, udf_func(col(in_col)))
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
            df = df.withColumn(c.replace("item", "category"), cat_udf(col(c)))
        return FeatureTable(df)


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
        Loads Parquet files, returning the result as a `StringIndex`.

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

    def _clone(self, df):
        return StringIndex(df, self.col_name)

    def write_parquet(self, path, mode="overwrite"):
        """
        Write StringIndex to Parquet file

        :param path: str. The path to the `folder` of the Parquet file. Note that the col_name
               will be used as basename of the Parquet file.
        :param mode: str. `append`, `overwrite`, `error` or `ignore`. `append`: Append contents
               of this StringIndex to existing data. `overwrite`: Overwrite existing data.
               `error`: Throw an exception if data already exists. `ignore`: Silently ignore this
               operation if data already exists.
        """
        path = path + "/" + self.col_name + ".parquet"
        self.df.write.parquet(path, mode=mode)
