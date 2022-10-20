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

import com.intel.analytics.bigdl.friesian.nearline.recall.RecallInitializer;
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils;
import com.intel.analytics.bigdl.friesian.serving.recall.IndexService;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.spark.sql.functions.col;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class RecallInitializerTest {
    static private final Logger logger = LogManager.getLogger(RecallInitializerTest.class.getName());

    @Test
    public void testInitialization() throws IOException, InterruptedException {
        URL resource = getClass().getClassLoader().getResource("testConfig/config_recall_init.yaml");
        assertNotNull(resource);
        RecallInitializer.main(new String[]{"-c", resource.getPath()});
        // you can get initialDataPath file from friesian-serving.tar.gz

        SparkSession sparkSession = SparkSession.builder().getOrCreate();
        Dataset<Row> dataset = sparkSession.read().parquet(NearlineUtils.helper().getInitialDataPath());

        dataset = dataset.select(col(NearlineUtils.helper().getItemIDColumn()),
                col(NearlineUtils.helper().getItemEmbeddingColumn())).distinct();

        IndexService indexService = new IndexService(NearlineUtils.helper().getIndexDim());
        indexService.load(NearlineUtils.helper().getIndexPath());

        assertEquals(dataset.count(), indexService.getNTotal());
        Map<Integer, float[]> vectorMap = new HashMap<>();
        List<Row> rows = dataset.collectAsList();
        rows.forEach(
                row -> vectorMap.put(
                        Integer.parseInt(row.get(0).toString()),
                        TestUtils.toFloatArray(row.get(1))));
        rows.forEach(row -> {
            float[] floatArr = TestUtils.toFloatArray(row.get(1));
            assertNotNull(vectorMap.get(indexService.search(IndexService.vectorToFloatArray(floatArr), 1)[0]));

            // assertArrayEquals(vectorMap.get(Integer.parseInt(row.get(0).toString())),
            //         vectorMap.get(indexService.search(IndexService.vectorToFloatArray(floatArr), 1)[0]));

            // assertEquals(indexService.search(IndexService.vectorToFloatArray(floatArr), 1)[0], row.getInt(0));
            // When multi vector have same value, may get unexpected id
        });
    }
}
