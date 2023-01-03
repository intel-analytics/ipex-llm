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

import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.types import IntegerType, DataType, StructType, StringType, StructField
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca.data import SparkXShards
from bigdl.orca import OrcaContext
from pyspark.ml.feature import MinMaxScaler as SparkMinMaxScaler
from pyspark.ml.feature import StandardScaler as SparkStandardScaler
from pyspark.ml.feature import VectorAssembler as SparkVectorAssembler
from pyspark.ml import Pipeline as SparkPipeline

from bigdl.orca.data.utils import *
import uuid

from typing import TYPE_CHECKING, Tuple, Union, List
if TYPE_CHECKING:
    from bigdl.orca.data.shard import SparkXShards
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame


# TODO: ADD UT
class StringIndexer:
    def __init__(self, inputCol: Union[str, List[str]]) -> None:
        self.inputCol = inputCol
        self.indices = None  # type: Union[StringIndex, List[StringIndex], None]

    def setInputCol(self, inputCol: Union[str, List[str]]) -> None:
        self.inputCol = inputCol

    def fit_transform(self, shard: "SparkXShards") -> "SparkXShards":
        df = shard.to_spark_df()
        indexedData, self.indices = self.category_encode(df, self.inputCol)
        data_shards = spark_df_to_pd_sparkxshards(indexedData)
        return data_shards

    def transform(self, shard: "SparkXShards") -> "SparkXShards":
        invalidInputError(self.indices, "Please call fit_transform first")
        df = shard.to_spark_df()
        indexedData = self.encode_string(df, self.inputCol, self.indices)  # type:ignore
        data_shards = spark_df_to_pd_sparkxshards(indexedData)
        return data_shards

    def gen_string_idx(self,
                       df: "SparkDataFrame",
                       columns: Union[str, List[str]],
                       freq_limit: Optional[Union[int, Dict[str, int]]]=None,
                       order_by_freq: bool=False,
                       do_split: bool=False,
                       sep: str=',') -> Union["StringIndex", List["StringIndex"]]:
        """
        Generate unique index value of categorical features. The resulting index would
        start from 1 with 0 reserved for unknown features.

        :param df: a pyspark.sql.dataframe.DataFrame to be processed.
        :param columns: str, dict or a list of str, dict, target column(s) to generate StringIndex.
         dict is a mapping of source column names -> target column name if needs to combine multiple
         source columns to generate index.
         For example: {'src_cols':['a_user', 'b_user'], 'col_name':'user'}.
        :param freq_limit: int, dict or None. Categories with a count/frequency below freq_limit
               will be omitted from the encoding. Can be represented as either an integer,
               dict. For instance, 15, {'col_4': 10, 'col_5': 2} etc. Default is None,
               and in this case all the categories that appear will be encoded.
        :param order_by_freq: boolean, whether the result StringIndex will assign smaller indices
               to values with more frequencies. Default is False and in this case frequency order
               may not be preserved when assigning indices.
        :param do_split: bool, whether need to split column value to array to generate index.
        Default is False.
        :param sep: str, a string representing a regular expression to split a column value.
         Default is ','.

        :return: A StringIndex or a list of StringIndex.
        """
        if columns is None:
            invalidInputError(False,
                              "columns should be str or a list of str, but got None.")
        is_single_column = False
        if not isinstance(columns, list):
            is_single_column = True
            columns = [columns]
        src_columns = []
        for c in columns:
            if isinstance(c, dict):
                if 'src_cols' in c:
                    src_columns.extend(c['src_cols'])
            else:
                src_columns.append(c)
        check_col_exists(df, src_columns)
        if freq_limit:
            if isinstance(freq_limit, int):
                freq_limit = str(freq_limit)  # type:ignore
            elif isinstance(freq_limit, dict):
                freq_limit = ",".join(str(k) + ":" + str(v)  # type:ignore
                                      for k, v in freq_limit.items())
            else:
                invalidInputError(False,
                                  "freq_limit only supports int, dict or None, but get " +
                                  freq_limit.__class__.__name__)
        out_columns = []
        simple_columns = []  # type: ignore
        df_id_list = []
        for c in columns:
            # union column
            if isinstance(c, dict):
                if 'src_cols' in c:
                    src_cols = c['src_cols']
                else:
                    invalidInputError(False,
                                      "Union columns must has argument 'src_cols'")
                if 'col_name' in c:
                    col_name = c['col_name']
                else:
                    col_name = src_cols[0] + '_union'
                # process simple columns
                if simple_columns:
                    simple_df_id_list = generate_string_idx(df, simple_columns,
                                                            freq_limit, order_by_freq)
                    df_id_list.extend(simple_df_id_list)
                    simple_columns = []
                # process union columns
                for i, src_c in enumerate(src_cols):
                    if i == 0:
                        dict_df = df.select(F.col(src_c).alias(col_name))
                    else:
                        dict_df = dict_df.union(df.select(F.col(src_c).alias(col_name)))
                union_id_list = generate_string_idx(dict_df, [col_name],
                                                    freq_limit, order_by_freq)
                df_id_list.extend(union_id_list)
                out_columns.append(col_name)
            # single column
            else:
                if do_split:
                    dict_df = df.select(F.col(c))
                    dict_df = dict_df.withColumn(c, F.explode(F.split(c, sep)))
                    split_id_list = generate_string_idx(dict_df, [c], freq_limit, order_by_freq)
                    df_id_list.extend(split_id_list)
                    out_columns.append(c)
                else:
                    simple_columns.append(c)
                    out_columns.append(c)
        if simple_columns:
            simple_df_id_list = generate_string_idx(df, simple_columns,
                                                    freq_limit, order_by_freq)
            df_id_list.extend(simple_df_id_list)

        string_idx_list = list(map(lambda x: StringIndex(x[0], x[1]),
                                   zip(df_id_list, out_columns)))

        # If input is a single column (not a list), then the output would be a single StringIndex.
        if len(string_idx_list) == 1 and is_single_column:
            return string_idx_list[0]
        else:
            return string_idx_list

    def category_encode(self,
                        df: "SparkDataFrame",
                        columns: Union[str, List[str]],
                        freq_limit: Optional[Union[int, Dict[str, int]]]=None,
                        order_by_freq: bool=False,
                        do_split: bool=False,
                        sep: str=',',
                        sort_for_array: bool=False,
                        keep_most_frequent: bool=False,
                        broadcast: bool=True) -> Tuple["SparkDataFrame",
                                                       Union["StringIndex", List["StringIndex"]]]:
        """
        Category encode the given columns.

        :param df: a pyspark.sql.dataframe.DataFrame to be processed.
        :param columns: str or a list of str, target columns to encode from string to index.
        :param freq_limit: int, dict or None. Categories with a count/frequency below freq_limit
               will be omitted from the encoding. Can be represented as either an integer,
               dict. For instance, 15, {'col_4': 10, 'col_5': 2} etc. Default is None,
               and in this case all the categories that appear will be encoded.
        :param order_by_freq: boolean, whether the result StringIndex will assign smaller indices
               to values with more frequencies. Default is False and in this case frequency order
               may not be preserved when assigning indices.
        :param do_split: bool, whether need to split column value to array to encode string.
               Default is False.
        :param sep: str, a string representing a regular expression to split a column value.
               Default is ','.
        :param sort_for_array: bool, whether need to sort array columns. Default is False.
        :param keep_most_frequent: bool, whether need to keep most frequent value as the
               column value. Default is False.
        :param broadcast: bool, whether need to broadcast index when encode string.
               Default is True.

        :return: A tuple of a new FeatureTable which transforms categorical features into unique
                 integer values, and a list of StringIndex for the mapping.
        """
        indices = self.gen_string_idx(df, columns, freq_limit=freq_limit,
                                      order_by_freq=order_by_freq,
                                      do_split=do_split, sep=sep)
        return self.encode_string(df, columns, indices, do_split=do_split, sep=sep,
                                  sort_for_array=sort_for_array,
                                  keep_most_frequent=keep_most_frequent,
                                  broadcast=broadcast), indices

    def encode_string(self,
                      df: "SparkDataFrame",
                      columns: Union[str, List[str]],
                      indices: Union["StringIndex", List["StringIndex"]],
                      broadcast: bool=True,
                      do_split: bool=False,
                      sep: str=',',
                      sort_for_array: bool=False,
                      keep_most_frequent: bool=False) -> "SparkDataFrame":
        """
        Encode columns with provided list of StringIndex. Unknown string will be
        None after the encoding and you may need to fillna with 0.

        :param columns: str or a list of str, the target columns to be encoded.
        :param indices: StringIndex or a list of StringIndex, StringIndexes of target columns.
               The StringIndex should at least have two columns: id and the corresponding
               categorical column.
               Or it can be a dict or a list of dicts. In this case,
               the keys of the dict should be within the categorical column
               and the values are the target ids to be encoded.
        :param broadcast: bool, whether need to broadcast index when encode string.
               Default is True.
        :param do_split: bool, whether need to split column value to array to encode string.
               Default is False.
        :param sep: str, a string representing a regular expression to split a column value.
               Default is ','.
        :param sort_for_array: bool, whether need to sort array columns. Default is False.
        :param keep_most_frequent: bool, whether need to keep most frequent value as the
               column value. Default is False.

        :return: A new FeatureTable which transforms categorical features into unique integer
                 values with provided StringIndexes.
        """
        if not isinstance(columns, list):
            columns = [columns]
        if not isinstance(indices, list):
            indices = [indices]
        invalidInputError(len(columns) == len(indices),
                          "columns len doesn't match indices lenngth")
        if isinstance(indices[0], dict):
            indices = list(map(lambda x: StringIndex.from_dict(x[1], columns[x[0]]),  # type:ignore
                               enumerate(indices)))
        data_df = df
        for i in range(len(columns)):
            index_tbl = indices[i]
            col_name = columns[i]
            if broadcast:
                index_tbl.broadcast()
            if not do_split:
                data_df = data_df.join(index_tbl.df, col_name, how="left") \
                    .drop(col_name).withColumnRenamed("id", col_name)
            else:
                data_df = data_df.withColumn('row_id', F.monotonically_increasing_id())
                tmp_df = data_df.select('row_id', col_name) \
                    .withColumn(col_name, F.explode(F.split(F.col(col_name), sep)))
                tmp_df = tmp_df.join(index_tbl.df, col_name, how="left") \
                    .filter(F.col("id").isNotNull())
                tmp_df = tmp_df.select('row_id', F.col("id"))
                if keep_most_frequent:
                    tmp_df = tmp_df.groupby('row_id') \
                        .agg(F.array_sort(F.collect_list(F.col("id")))
                             .getItem(0).alias("id"))
                elif sort_for_array:
                    tmp_df = tmp_df.groupby('row_id') \
                        .agg(F.array_sort(F.collect_list(F.col("id"))).alias("id"))
                else:
                    tmp_df = tmp_df.groupby('row_id') \
                        .agg(F.collect_list(F.col("id")).alias("id"))
                data_df = data_df.join(tmp_df, 'row_id', 'left') \
                    .drop('row_id').drop(col_name).withColumnRenamed("id", col_name)

        return data_df


class StringIndex:
    def __init__(self, df: "SparkDataFrame", col_name: str) -> None:
        cols = df.columns
        invalidInputError(len(cols) >= 2,
                          "StringIndex should have >= 2 columns: col_name, id and other columns")
        invalidInputError("id" in cols, "id should be a column of the DataFrame")
        invalidInputError(col_name in cols, col_name + " should be a column of the DataFrame")
        self.col_name = col_name
        self.df = df

    def broadcast(self) -> None:
        from pyspark.sql.functions import broadcast
        self.df = broadcast(self.df)

    @classmethod
    def from_dict(cls, indices: Dict[str, int], col_name: str) -> "StringIndex":
        """
        Create the StringIndex from a dict of indices.

        :param indices: dict, the key is the categorical column, and the value is the
               corresponding index. We assume that the key is a str and the value
               is a int.
        :param col_name: str. The column name of the categorical column.

        :return: A StringIndex.
        """
        spark = OrcaContext.get_spark_session()
        if not isinstance(indices, dict):
            invalidInputError(False,
                              'indices should be dict, but get ' + indices.__class__.__name__)
        if not col_name:
            invalidInputError(False,
                              'col_name should be str, but get None')
        if not isinstance(col_name, str):
            invalidInputError(False,
                              'col_name should be str, but get ' + col_name.__class__.__name__)
        new_indices = map(lambda x: {col_name: x[0], 'id': x[1]}, indices.items())
        schema = StructType([StructField(col_name, StringType(), False),
                             StructField("id", IntegerType(), False)])
        df = spark.createDataFrame((Row(**x) for x in new_indices), schema=schema)
        return cls(df, col_name)

    def to_dict(self) -> Dict[str, int]:
        """
        Convert the StringIndex to a dict, with the categorical features as keys and indices
        as values.
        Note that you may only call this if the StringIndex is small.

        :return: A dict for the mapping from string to index.
        """
        cols = self.df.columns
        index_id = cols.index("id")
        col_id = cols.index(self.col_name)
        rows = self.df.collect()
        res_dict = {}
        for row in rows:
            res_dict[row[col_id]] = row[index_id]
        return res_dict


# TODO: ADD UT
class MinMaxScaler:
    def __init__(self,
                 min: float=0.0,
                 max: float=1.0,
                 inputCol: Optional[Union[str, List[str]]]=None,
                 outputCol: Optional[str]=None):
        self.min = min
        self.max = max
        self.inputCol = [inputCol] if isinstance(inputCol, str) else inputCol
        self.outputCol = outputCol
        self.scaler = None  # type: Optional[SparkPipeline]
        self.scalerModel = None
        if inputCol:
            self.__createScaler__()

    def __createScaler__(self) -> None:
        invalidInputError(self.inputCol, "inputColumn cannot be empty")
        invalidInputError(self.outputCol, "outputColumn cannot be empty")

        self.vecOutputCol = str(uuid.uuid1()) + "x_vec"
        assembler = SparkVectorAssembler(inputCols=self.inputCol,  # type:ignore
                                         outputCol=self.vecOutputCol)
        scaler = SparkMinMaxScaler(min=self.min, max=self.max,
                                   inputCol=self.vecOutputCol,
                                   outputCol=self.outputCol)  # type:ignore
        self.scaler = SparkPipeline(stages=[assembler, scaler])

    def setInputOutputCol(self,
                          inputCol: Union[str, List[str]],
                          outputCol: str) -> None:
        self.inputCol = [inputCol] if isinstance(inputCol, str) else inputCol
        self.outputCol = outputCol
        self.__createScaler__()

    def fit_transform(self, shard: "SparkXShards") -> "SparkXShards":
        df = shard.to_spark_df()
        self.scalerModel = self.scaler.fit(df)  # type: ignore
        scaledData = self.scalerModel.transform(df)  # type: ignore
        scaledData = scaledData.drop(self.vecOutputCol)
        data_shards = spark_df_to_pd_sparkxshards(scaledData)
        return data_shards

    def transform(self, shard: "SparkXShards") -> "SparkXShards":
        invalidInputError(self.scalerModel, "Please call fit_transform first")
        df = shard.to_spark_df()
        scaledData = self.scalerModel.transform(df)  # type: ignore
        scaledData = scaledData.drop(self.vecOutputCol)
        data_shards = spark_df_to_pd_sparkxshards(scaledData)
        return data_shards


class StandardScaler:
    def __init__(self,
                 withMean: bool = False,
                 withStd: bool = True,
                 inputCol: Optional[Union[str, List[str]]]=None,
                 outputCol: Optional[str]=None):
        self.withMean = withMean
        self.withStd = withStd
        self.inputCol = [inputCol] if isinstance(inputCol, str) else inputCol
        self.outputCol = outputCol
        self.scaler = None
        self.scalerModel = None
        if inputCol:
            self.__createScaler__()

    def __createScaler__(self):
        invalidInputError(self.inputCol, "inputColumn cannot be empty")
        invalidInputError(self.outputCol, "outputColumn cannot be empty")

        self.vecOutputCol = str(uuid.uuid1()) + "x_vec"
        assembler = SparkVectorAssembler(inputCols=self.inputCol, outputCol=self.vecOutputCol)
        scaler = SparkStandardScaler(withMean=self.withMean, withStd=self.withStd,
                                     inputCol=self.vecOutputCol, outputCol=self.outputCol)
        self.scaler = SparkPipeline(stages=[assembler, scaler])

    def setInputOutputCol(self,
                          inputCol: Union[str, List[str]],
                          outputCol: str) -> None:
        self.inputCol = [inputCol] if isinstance(inputCol, str) else inputCol
        self.outputCol = outputCol
        self.__createScaler__()

    def fit_transform(self, shard: "SparkXShards") -> "SparkXShards":
        df = shard.to_spark_df()
        self.scalerModel = self.scaler.fit(df)  # type: ignore
        scaledData = self.scalerModel.transform(df)  # type: ignore
        scaledData = scaledData.drop(self.vecOutputCol)
        data_shards = spark_df_to_pd_sparkxshards(scaledData)
        return data_shards

    def transform(self, shard: "SparkXShards") -> "SparkXShards":
        invalidInputError(self.scalerModel, "Please call fit_transform first")
        df = shard.to_spark_df()
        scaledData = self.scalerModel.transform(df)  # type: ignore
        scaledData = scaledData.drop(self.vecOutputCol)
        data_shards = spark_df_to_pd_sparkxshards(scaledData)
        return data_shards
