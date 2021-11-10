# connect sqlite3
import time
import parse
from pyspark.sql import SparkSession

# create parser object
parser = argparse.ArgumentParser(description='Process e2e programming')
parser.add_argument('--db_path', type=str, help='Please make sure e2e database path passed in.')
args = parser.parse_args()

db_path = args.db_path

inited = time.time()*1000
print("PERF inited", inited)

spark = SparkSession.builder.getOrCreate()

spark_inited = time.time()*1000
print("PERF spark_inited", spark_inited)

alice = spark.read.jdbc('jdbc:sqlite:{}'.format(db_path), 'alice', column="a1169", lowerBound=0, upperBound=1000, numPartitions=128)
bob = spark.read.jdbc('jdbc:sqlite:{}'.format(db_path), 'bob')
db_read = time.time()*1000
print("PERF db_read", db_read)

alice.createOrReplaceTempView("alice")
bob.createOrReplaceTempView("bob")
viewed = time.time()*1000
print("PERF viewed", viewed)

# a.
result1 = bob.filter(bob.c_id == 'ab').join(alice, alice.id == bob.id).filter('c1028 is not null').groupBy('c1028').count().sort('c1028')
result1.show(n=1000, truncate = False)
result1ed = time.time()*1000
print("PERF result1ed", result1ed)
print("INFO result1: ", str(result1))
# result count
print(result1.count())

# b.
result2 = alice.filter('a1169 in (151, 152)').union(alice.filter('b1209 in (220, 330)')).union(alice.filter('c1034 in (422, 63)')).distinct().count()
result2ed = time.time()*1000
print("PERF result2ed", result2ed)
print("INFO this is results count: ", result2)
