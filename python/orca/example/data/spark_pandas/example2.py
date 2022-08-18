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

    file_path = options.file_path
    data_shard = bigdl.orca.data.pandas.read_csv(file_path)
    data = data_shard.collect()

    # drop
    def trans_func(df):
        df = df.drop(['PassengerId'], axis=1)
        return df
    transformed_data_shard1 = data_shard.transform_shard(trans_func)
    trans_data3 = transformed_data_shard1.collect()

    # fillna, apply, replace, map
    def trans_func(df):
        df['Cabin'] = df['Cabin'].fillna('X')
        df['Cabin'] = df['Cabin'].apply(lambda x: str(x)[0])
        df['Cabin'] = df['Cabin'].replace(['A', 'D', 'E', 'T'], 'M')
        df['Cabin'] = df['Cabin'].replace(['B', 'C'], 'H')
        df['Cabin'] = df['Cabin'].replace(['F', 'G'], 'L')
        df['Cabin'] = df['Cabin'].map({'X': 0, 'L': 1, 'M': 2, 'H': 3})
        df['Cabin'] = df['Cabin'].astype(int)
        return df
    transformed_data_shard2 = transformed_data_shard1.transform_shard(trans_func)
    trans_data4 = transformed_data_shard2.collect()

    # astype, loc
    def trans_func(data):
        data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})
        data['Pclass'] = data['Pclass'].map({1: 3, 2: 2, 3: 1}).astype(int)
        data.loc[data['Sex'] == 0, 'SexByPclass'] = data.loc[data['Sex'] == 0, 'Pclass']
        data.loc[data['Sex'] == 1, 'SexByPclass'] = data.loc[data['Sex'] == 1, 'Pclass'] + 3
        return data
    transformed_data_shard3 = transformed_data_shard2.transform_shard(trans_func)
    trans_data5 = transformed_data_shard3.collect()
