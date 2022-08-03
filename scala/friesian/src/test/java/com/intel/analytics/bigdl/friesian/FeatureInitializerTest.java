package com.intel.analytics.bigdl.friesian;

import com.intel.analytics.bigdl.friesian.nearline.feature.FeatureInitializer;
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils;
import com.intel.analytics.bigdl.friesian.serving.feature.utils.LettuceUtils;
import com.intel.analytics.bigdl.friesian.serving.feature.utils.RedisType;
import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Test;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.io.IOException;
import java.util.Arrays;
import java.util.Base64;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class FeatureInitializerTest {


    public static Seq<String> convertListToSeq(String[] inputArray) {
        return JavaConverters.asScalaIteratorConverter(Arrays.asList(inputArray).iterator()).asScala().toSeq();
    }

    private String generateID(String keyPrefix, String ID) {
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

    private void checkRedisRecord(LettuceUtils redis, String keyPrefix, Dataset<Row> dataset) {
        dataset.collectAsList().forEach((row) -> {
            Object[] rowData = (Object[]) EncodeUtils.bytesToObj(
                    Base64.getDecoder().decode(
                            redis.getSync().getdel(generateID(keyPrefix, row.get(0).toString()))));
            for (int i = 0; i < Objects.requireNonNull(rowData).length; i++) {
                assertEquals(rowData[i].toString(), row.get(1 + i).toString());
            }
//            assertNotEquals(redis.getSync().getdel(generateID(keyPrefix, row.get(0).toString())), null);
        });
    }

    @Test
    public void testInitialization() throws IOException, InterruptedException {
        String configPath = "/home/xingyuan/projects/serving/BigDL/scala/friesian/src/test/resources/nearlineConfig/config_feature_vec.yaml";
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
            checkRedisRecord(redis, "user", dataset);
        }

        if (NearlineUtils.helper().initialItemDataPath() != null) {
            assertEquals(redis.getSchema("item"), String.join(",", NearlineUtils.helper().itemFeatureColArr()));
            SparkSession sparkSession = SparkSession.builder().getOrCreate();
            Dataset<Row> dataset = sparkSession.read().parquet(NearlineUtils.helper().initialItemDataPath());
            dataset = dataset.select(NearlineUtils.helper().itemIDColumn(),
                    convertListToSeq(NearlineUtils.helper().itemFeatureColArr()));
            checkRedisRecord(redis, "item", dataset);
        }
        System.out.println("Finish Test");
    }
}
