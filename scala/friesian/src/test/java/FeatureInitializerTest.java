import com.intel.analytics.bigdl.friesian.nearline.feature.FeatureInitializer;
import com.intel.analytics.bigdl.friesian.nearline.feature.FeatureNearlineUtils;
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils;
import com.intel.analytics.bigdl.friesian.serving.feature.utils.LettuceUtils;
import com.intel.analytics.bigdl.friesian.serving.feature.utils.RedisType;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Test;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

public class FeatureInitializerTest {


    public static Seq<String> convertListToSeq(String[] inputArray) {
        return JavaConverters.asScalaIteratorConverter(Arrays.asList(inputArray).iterator()).asScala().toSeq();
    }

    String generateID(String keyPrefix, String ID) {
        String redisKeyPrefix = NearlineUtils.helper().redisKeyPrefix();
        RedisType redisType = NearlineUtils.helper().redisTypeEnum();
        int itemSlotType = NearlineUtils.helper().itemSlotType();
        if (redisType == RedisType.CLUSTER && keyPrefix.equals("item") && itemSlotType != 0) {
            if (itemSlotType == 1) {
                return redisKeyPrefix + "{" + keyPrefix + "}:" + ID;
            } else {
                return "{" + redisKeyPrefix + keyPrefix + ID.charAt(ID.length() - 1) + "}:" + ID;
            }
        } else {
            return redisKeyPrefix + keyPrefix + ":" + ID;
        }
    }

    @Test
    void testInitialization() throws IOException, InterruptedException {
        String configPath = "/home/xingyuan/projects/serving/BigDL/scala/friesian/src/test/resources/nearlineConfig/config_feature.yaml";
        FeatureInitializer.main(new String[]{"-c", configPath});
        // you can get initialDataPath file from friesian-serving.tar.gz

        LettuceUtils redis = LettuceUtils.getInstance(NearlineUtils.helper().redisTypeEnum(),
                NearlineUtils.helper().redisHostPort(), NearlineUtils.helper().getRedisKeyPrefix(),
                NearlineUtils.helper().redisSentinelMasterURL(), NearlineUtils.helper().redisSentinelMasterName(),
                NearlineUtils.helper().itemSlotType());

        if (NearlineUtils.helper().initialUserDataPath() != null) {
            assertEquals(redis.getSchema("user"), String.join(",", NearlineUtils.helper().userFeatureColArr()));
            SparkSession sparkSession = SparkSession.builder().getOrCreate();
            Dataset<Row> dataset = sparkSession.read().parquet(NearlineUtils.helper().initialUserDataPath());
            dataset = dataset.select(NearlineUtils.helper().userIDColumn(),
                    convertListToSeq(NearlineUtils.helper().userFeatureColArr()));
            dataset.collectAsList().forEach((row) -> {
                assertEquals(redis.get(generateID("user", row.get(0).toString())),
                        FeatureNearlineUtils.encodeRow(row)[1]);
                assertNotEquals(redis.getSync().getdel(generateID("user", row.get(0).toString())), null);
            });
        }

        if (NearlineUtils.helper().initialItemDataPath() != null) {
            assertEquals(redis.getSchema("item"), String.join(",", NearlineUtils.helper().itemFeatureColArr()));
            SparkSession sparkSession = SparkSession.builder().getOrCreate();
            Dataset<Row> dataset = sparkSession.read().parquet(NearlineUtils.helper().initialItemDataPath());
            dataset = dataset.select(NearlineUtils.helper().itemIDColumn(),
                    convertListToSeq(NearlineUtils.helper().itemFeatureColArr()));
            dataset.collectAsList().forEach((row) -> {
                assertEquals(redis.get(generateID("item", row.get(0).toString())),
                        FeatureNearlineUtils.encodeRow(row)[1]);
                assertNotEquals(redis.getSync().getdel(generateID("item", row.get(0).toString())), null);
            });
        }
        System.out.println("Finish Test");
    }
}
