import random
import os
from argparse import ArgumentParser
from pyspark.sql.types import StructType, StructField, StringType, LongType, BooleanType
from bigdl.friesian.feature import FeatureTable
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext

id_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8",
           "9", "A", "B", "C", "D", "E", "F", "G", "H",
           "I", "J", "K", "L", "M", "N", "O", "P", "Q",
           "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
media_list = ["Photo", "Video", "GIF"]
tweet_list = ["Retweet", "Quote", "TopLevel"]
language_list = ['DE8A3755FCEDC549A408D7B1EB1A2C9F', '9D831A0F3603A54732CCBDBF291D17B7', 'D922D8FEA3EFAD3200455120B75BCEB8', 'F9D8F1DB5A398E1225A2C42E34A51DF6', 'A5CFB818D79497B482B7225887DBD3AD', '2573A3CF633EBE6932A1E1010D5CD213', 'C1E99BF67DDA2227007DE8038FE32470', '477ED2ED930405BF1DBF13F9BF973434', '8729EBF694C3DAF61208A209C2A542C8', '10C6C994C2AD434F9D49D4BE9CFBC613', '89CE0912454AFE0A1B959569C37A5B8F', '914074E75CB398B5A2D81E1A51818CAA', '9A78FC330083E72BE0DD1EA92656F3B5', 'E84BE2C963852FB065EE827F41A0A304', '23686A079CA538645BF6118A1EF51C8B', '7D11A7AA105DAB4D6799AF863369DB9C', '9FCF19233EAD65EA6E32C2E6DC03A444', '313ECD3A1E5BB07406E4249475C2D6D6', '159541FA269CA8A9CDB93658CAEC4CA2', 'CDE47D81F953D800F760F1DE8AA754BA', '2F548E5BE0D7F678E72DDE31DFBEF8E7', 'F33767F7D7080003F403FDAB34FEB755', '5F152815982885A996841493F2757D91', 'B0FA488F2911701DD8EC5B1EA5E322D8', '5B210378BE9FFA3C90818C43B29B466B', '1F73BB863A39DB62B4A55B7E558DB1E8', '3EA57373381A56822CBBC736169D0145', '00304D7356D6C64481190D708D8F739C', 'BAC6A3C2E18C26A77C99B41ECE1C738D', 'DA13A5C3763C212D9D68FC69102DE5E5', '7E18F69967284BB0601E88A114B8F7A9', '5A0759FB938B1D9B1E08B7A3A14F1042',
                 'A3E4360031A7E05E9279F4D504EE18DD', '3AB05D6A4045A6C37D3E4566CFDFFE26', '5B6973BEB05212E396F3F2DC6A31B71C', 'BF04E736C599E9DE22F39F1DC157E1F1', '838A92D9F7EB57FB4A8B0C953A80C7EB', 'C41F6D723AB5D14716D856DF9C000DED', '37342508F52BF4B62CCE3BA25460F9EB', 'DC5C9FB3F0B3B740BAEE4F6049C2C7F1', '310ECD7D1E42216E3C1B31EFDDFC72A7', '99CA116BF6AA65D70F3C78BEBADC51F0', '3DF931B225B690508A63FD24133FA0E2', 'E7F038DE3EAD397AEC9193686C911677', '3228B1FB4BC92E81EF2FE35BDA86C540', 'F70598172AC4514B1E6818EA361AD580', '59BE899EB83AAA19878738040F6828F0', 'B4DC2F82961F1263E90DF7A942CCE0B2', '488B32D24BD4BB44172EB981C1BCA6FA', 'B8B04128918BBF54E2E178BFF1ABA833', '678E280656F6A0C0C23D5DFD46B85C14', '4B55C45CD308068E4D0913DEF1043AD6', '440116720BC3A7957E216A77EE5C18CF', '0BB2C843174730BA7D958C98B763A797', '7F4FAB1EB12CD95EDCD9DB2A6634EFCE', '105008E45831ADE8AF1DB888319F422A', '6B90065EA806B8523C0A6E56D7A961B2', '41776FB50B812A6775C2F8DEC92A9779', 'D7C16BC3C9A5A633D6A3043A567C95A6', '4CA37504EF8BA4352B03DCBA50E98A45', '7168CE9B777B76E4069A538DC5F28B6F', 'E6936751CBF4F921F7DE1AEF33A16ED0', '8C64085F46CD49FA5C80E72A35845185', '6744F8519308FD72D8C47BD45186303C', 'CF304ED3CFC1ADD26720B97B39900FFD']


schema = StructType(
    [StructField("test_tokens", StringType(), True),
     StructField("hashtags", StringType(), True),
     StructField("tweet_id", StringType(), True),
     StructField("present_media", StringType(), True),
     StructField("present_links", StringType(), True),
     StructField("present_domains", StringType(), True),
     StructField("tweet_type", StringType(), True),
     StructField("language", StringType(), True),
     StructField("tweet_timestamp", LongType(), True),
     StructField("engaged_with_user_id", StringType(), True),
     StructField("engaged_with_user_follower_count", LongType(), True),
     StructField("engaged_with_user_following_count", LongType(), True),
     StructField("engaged_with_user_is_verified", BooleanType(), True),
     StructField("engaged_with_user_account_creation", LongType(), True),
     StructField("enaging_user_id", StringType(), True),
     StructField("enaging_user_follower_count", LongType(), True),
     StructField("enaging_user_following_count", LongType(), True),
     StructField("enaging_user_is_verified", BooleanType(), True),
     StructField("enaging_user_account_creation", LongType(), True),
     StructField("engagee_follows_engager", StringType(), True),
     StructField("reply_timestamp", LongType(), True),
     StructField("retweet_timestamp", LongType(), True),
     StructField("retweet_with_comment_timestamp", LongType(), True),
     StructField("like_timestamp", LongType(), True)])


def _parse_args():
    parser = ArgumentParser(description="Generate random dataset for demo")
    parser.add_argument('num_samples', type=int,
                        help='The number of samples')
    parser.add_argument('--memory', type=str, default="32g",
                        help='The executor memory')
    parser.add_argument('output_path', type=str,
                        help='The path for output dataset')
    args = parser.parse_args()
    return args


def generate_record(random_seed):
    random.seed(random_seed)
    test_tokens = "\t".join([str(random.randint(1, 1000))
                            for i in range(random.randint(1, 10))])
    hashtags = "\t".join(["".join(random.choices(id_list, k=32))
                          for i in range(random.randint(0, 50))])
    tweet_id = "".join(random.choices(id_list, k=32))
    present_media = "\t".join(random.choices(
        media_list, k=random.randint(0, 9)))
    present_links = "\t".join(["".join(random.choices(id_list, k=32))
                               for i in range(random.randint(0, 10))])
    present_domains = "\t".join(["".join(random.choices(id_list, k=32))
                                for i in range(random.randint(0, 10))])
    tweet_type = random.choices(tweet_list)[0]
    language = random.choices(language_list)[0]
    tweet_timestamp = random.randint(946656000, 1609430400)
    engaged_with_user_id = "".join(random.choices(id_list, k=32))
    engaged_with_user_follower_count = random.randint(0, 10000)
    engaged_with_user_following_count = random.randint(0, 10000)
    engaged_with_user_is_verified = bool(random.getrandbits(1))
    engaged_with_user_account_creation = random.randint(946656000, 1609430400)
    enaging_user_id = "".join(random.choices(id_list, k=32))
    enaging_user_follower_count = random.randint(0, 10000)
    enaging_user_following_count = random.randint(0, 10000)
    enaging_user_is_verified = bool(random.getrandbits(1))
    enaging_user_account_creation = random.randint(946656000, 1609430400)
    engagee_follows_engager = bool(random.getrandbits(1))
    reply = bool(random.getrandbits(1))
    reply_timestamp = random.randint(946656000, 1609430400) if reply else None
    retweet = bool(random.getrandbits(1))
    retweet_timestamp = random.randint(
        946656000, 1609430400) if retweet else None
    comment = bool(random.getrandbits(1))
    retweet_with_comment_timestamp = random.randint(
        946656000, 1609430400) if comment else None
    like = bool(random.getrandbits(1))
    like_timestamp = random.randint(946656000, 1609430400) if like else None
    return (test_tokens, hashtags, tweet_id, present_media, present_links, present_domains,
            tweet_type, language, tweet_timestamp, engaged_with_user_id,
            engaged_with_user_follower_count, engaged_with_user_following_count,
            engaged_with_user_is_verified, engaged_with_user_account_creation,
            enaging_user_id, enaging_user_follower_count, enaging_user_following_count,
            enaging_user_is_verified, enaging_user_account_creation,
            engagee_follows_engager, reply_timestamp, retweet_timestamp,
            retweet_with_comment_timestamp, like_timestamp)


if __name__ == '__main__':
    args = _parse_args()

    OrcaContext.log_output = True

    executor_cores = 8
    num_executor = 4
    executor_memory = args.memory
    driver_cores = 4
    driver_memory = "12g"
    conf = {"spark.network.timeout": "10000000",
            "spark.sql.broadcastTimeout": "7200",
            "spark.sql.shuffle.partitions": "2000",
            "spark.locality.wait": "0s",
            "spark.sql.crossJoin.enabled": "true",
            "spark.task.cpus": "1",
            "spark.executor.heartbeatInterval": "200s",
            "spark.driver.maxResultSize": "40G",
            "spark.eventLog.enabled": "true",
            "spark.app.name": "recsys-dummy-data-generation",
            "spark.debug.maxToStringFields": "100"}
    sc = init_orca_context("yarn", cores=executor_cores,
                           num_nodes=num_executor, memory=executor_memory,
                           driver_cores=driver_cores, driver_memory=driver_memory,
                           conf=conf)
    spark = OrcaContext.get_spark_session()

    rdd = sc.parallelize(range(args.num_samples))
    dummy_data_rdd = rdd.map(generate_record)
    df = FeatureTable(spark.createDataFrame(dummy_data_rdd, schema))
    print(df.show(2))

    train_df, test_df = df.random_split([0.8, 0.2])
    print('train set size = ', train_df.size())
    print('test set size = ', test_df.size())

    train_df.write_parquet(os.path.join(args.output_path, 'train'))
    test_df.write_parquet(os.path.join(args.output_path, 'test'))
    print('Save data finished')

    stop_orca_context()
