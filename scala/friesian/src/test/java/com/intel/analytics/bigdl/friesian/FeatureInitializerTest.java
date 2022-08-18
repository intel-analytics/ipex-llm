/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.friesian;

import com.intel.analytics.bigdl.friesian.nearline.feature.FeatureInitializer;
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils;
import com.intel.analytics.bigdl.friesian.serving.feature.utils.LettuceUtils;
import com.intel.analytics.bigdl.friesian.serving.feature.utils.RedisType;
import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.net.URL;
import java.util.Base64;
import java.util.List;
import java.util.Objects;

import static com.intel.analytics.bigdl.friesian.JavaTestUtils.convertListToSeq;
import static com.intel.analytics.bigdl.friesian.JavaTestUtils.destroyLettuceUtilsInstance;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class FeatureInitializerTest {


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
        List<Row> rawDataList = dataset.collectAsList();
        List<String> redisDataList = redis.MGet(keyPrefix,
                rawDataList.stream().map(row -> row.get(0).toString()).toArray(String[]::new));
        for (int i = 0; i < rawDataList.size(); i++) {
            Object[] redisData = (Object[]) EncodeUtils.bytesToObj(
                    Base64.getDecoder().decode(redisDataList.get(i)));
            // compare each column data between redis decoded row and row data row
            for (int j = 0; j < Objects.requireNonNull(redisData).length; j++) {
                assertEquals(redisData[j].toString(), rawDataList.get(i).get(1 + j).toString());
            }
        }

        // test id generating code, del insert keys
        rawDataList.forEach((row) -> {
            Object[] rowData = (Object[]) EncodeUtils.bytesToObj(
                    Base64.getDecoder().decode(
                            redis.getSync().getdel(generateID(keyPrefix, row.get(0).toString()))));
            // compare each column data between redis decoded row and row data row
            for (int i = 0; i < Objects.requireNonNull(rowData).length; i++) {
                assertEquals(rowData[i].toString(), row.get(1 + i).toString());
            }
            // assertNotEquals(redis.getSync().getdel(generateID(keyPrefix, row.get(0).toString())), null);
        });
    }

    @ParameterizedTest
    @ValueSource(strings = {"testConfig/config_feature_init.yaml", "testConfig/config_feature_vec_init.yaml"})
    public void testFeatureInit(String configPath)
            throws IOException, InterruptedException, NoSuchFieldException, IllegalAccessException {
        URL resource = getClass().getClassLoader().getResource(configPath);
        assert resource != null;
        FeatureInitializer.main(new String[]{"-c", resource.getPath()});
        // you can get initialDataPath file from friesian-serving.tar.gz

        LettuceUtils redis = LettuceUtils.getInstance(NearlineUtils.helper().redisTypeEnum(),
                NearlineUtils.helper().redisHostPort(), NearlineUtils.helper().getRedisKeyPrefix(),
                NearlineUtils.helper().redisSentinelMasterURL(), NearlineUtils.helper().redisSentinelMasterName(),
                NearlineUtils.helper().itemSlotType());

        if (NearlineUtils.helper().initialUserDataPath() != null) {
            // check user scheme
            assertEquals(redis.getSync().getdel(NearlineUtils.helper().getRedisKeyPrefix() + "user"),
                    String.join(",", NearlineUtils.helper().userFeatureColArr()));
            SparkSession sparkSession = SparkSession.builder().getOrCreate();
            Dataset<Row> dataset = sparkSession.read().parquet(NearlineUtils.helper().initialUserDataPath());
            dataset = dataset.select(NearlineUtils.helper().userIDColumn(),
                    convertListToSeq(NearlineUtils.helper().userFeatureColArr()));
            checkRedisRecord(redis, "user", dataset);
        }

        if (NearlineUtils.helper().initialItemDataPath() != null) {
            assertEquals(redis.getSync().getdel(NearlineUtils.helper().getRedisKeyPrefix() + "item"),
                    String.join(",", NearlineUtils.helper().itemFeatureColArr()));
            SparkSession sparkSession = SparkSession.builder().getOrCreate();
            Dataset<Row> dataset = sparkSession.read().parquet(NearlineUtils.helper().initialItemDataPath());
            dataset = dataset.select(NearlineUtils.helper().itemIDColumn(),
                    convertListToSeq(NearlineUtils.helper().itemFeatureColArr()));
            checkRedisRecord(redis, "item", dataset);
        }
        System.out.println("Finish Test");
        destroyLettuceUtilsInstance();
    }
}
