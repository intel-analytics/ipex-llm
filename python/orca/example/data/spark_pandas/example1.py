import sys
from optparse import OptionParser
import bigdl.orca.data
import bigdl.orca.data.pandas
from bigdl.orca import OrcaContext

OrcaContext.pandas_read_backend = "pandas"

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--f", type=str, dest="file_path", help="The file path to be read")
    (options, args) = parser.parse_args(sys.argv)

    # read
    file_path = options.file_path
    data_shard = bigdl.orca.data.pandas.read_csv(file_path)
    data = data_shard.collect()
    df_shard = data[0]

    # fillna
    def trans_func(df):
        df = df.fillna(method='pad')
        return df
    transformed_data_shard = data_shard.transform_shard(trans_func)
    trans_data1 = transformed_data_shard.collect()

    # apply
    def trans_func(df):
        dic = {'A': 'Beijing', 'B': 'Shanghai', 'C': 'Guangzhou'}
        df['City_Category'] = df['City_Category'].apply(lambda x: dic[x])
        return df
    transformed_data_shard = data_shard.transform_shard(trans_func)
    trans_data2 = transformed_data_shard.collect()

    # map
    def trans_func(df):
        df = df['City_Category'].map({'A': 'Beijing', 'B': 'Shanghai', 'C': 'Guangzhou'})
        return df
    transformed_data_shard = data_shard.transform_shard(trans_func)
    trans_data = transformed_data_shard.collect()
