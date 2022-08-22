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

import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.friesian.nearline.feature.FeatureInitializer;
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils;
import com.intel.analytics.bigdl.friesian.serving.feature.FeatureServer;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureGrpc.FeatureBlockingStub;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto;
import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Base64;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

import static com.intel.analytics.bigdl.friesian.JavaTestUtils.convertListToSeq;
import static com.intel.analytics.bigdl.friesian.JavaTestUtils.destroyLettuceUtilsInstance;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;


public class FeatureServerTest {
    private static final Logger logger = LogManager.getLogger(FeatureServerTest.class.getName());
    private static FeatureBlockingStub featureBlockingStub;
    private static ManagedChannel channel;
    private static List<Row> itemRowList = null, userRowList = null;
    private static FeatureServer featureServer;

    /**
     * Sets up the test fixture.
     * (Called before every test case method.)
     */
    @BeforeAll
    public static void setUp() throws Exception {
        String configDir = Objects.requireNonNull(
                FeatureServer.class.getClassLoader().getResource("testConfig")).getPath();
        FeatureInitializer.main(new String[]{"-c", configDir + "/config_feature_init.yaml"});
        destroyLettuceUtilsInstance();

        SparkSession sparkSession = SparkSession.builder().getOrCreate();
        if (NearlineUtils.helper().initialUserDataPath() != null) {
            Dataset<Row> userDataset = sparkSession.read().parquet(NearlineUtils.helper().initialUserDataPath());
            userDataset = userDataset.select(
                    NearlineUtils.helper().userIDColumn(),
                    convertListToSeq(NearlineUtils.helper().userFeatureColArr()));
            userRowList = userDataset.collectAsList();
        }
        if (NearlineUtils.helper().initialItemDataPath() != null) {
            Dataset<Row> itemDataset = sparkSession.read().parquet(NearlineUtils.helper().initialItemDataPath());
            itemDataset = itemDataset.select(
                    NearlineUtils.helper().itemIDColumn(),
                    convertListToSeq(NearlineUtils.helper().itemFeatureColArr()));
            itemRowList = itemDataset.collectAsList();
        }

        featureServer = new FeatureServer(new String[]{"-c", configDir + "/config_feature_server.yaml"});
        featureServer.parseConfig();
        featureServer.build();
        featureServer.start();
        destroyLettuceUtilsInstance();

        channel = ManagedChannelBuilder.forAddress(
                "localhost", Utils.helper().getServicePort()).usePlaintext().build();
        featureBlockingStub = FeatureGrpc.newBlockingStub(channel);
    }

    /**
     * Tears down the test fixture.
     * (Called after every test case method.)
     */
    @AfterAll
    public static void tearDown() throws InterruptedException {
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        featureServer.stop();
    }


    @Test
    public void testGetUserFeatures() {
        FeatureProto.IDs.Builder builder = FeatureProto.IDs.newBuilder();
        for (Row row : userRowList) {
            builder.addID(row.getInt(0));
        }
        FeatureProto.Features features = null;
        try {
            features = featureBlockingStub.getUserFeatures(builder.build());
        } catch (StatusRuntimeException e) {
            logger.error("RPC failed: " + e.getStatus());
        }
        assertNotNull(features);
        List<String> featureList = features.getB64FeatureList();
        assertEquals(featureList.size(), userRowList.size());
        for (int i = 0; i < featureList.size(); i++) {
            Object[] featureArr = (Object[]) Objects.requireNonNull(
                    EncodeUtils.bytesToObj(Base64.getDecoder().decode(featureList.get(i))));
            for (int j = 0; j < featureArr.length; j++) {
                assertEquals(featureArr[j], userRowList.get(i).get(j + 1));
            }
        }
    }

    @Test
    public void testGetItemFeatures() {
        FeatureProto.IDs.Builder builder = FeatureProto.IDs.newBuilder();
        for (Row row : itemRowList) {
            builder.addID(row.getInt(0));
        }
        FeatureProto.Features features = null;
        try {
            features = featureBlockingStub.getItemFeatures(builder.build());
        } catch (StatusRuntimeException e) {
            logger.error("RPC failed: " + e.getStatus());
        }
        assertNotNull(features);
        List<String> featureList = features.getB64FeatureList();
        assertEquals(featureList.size(), itemRowList.size());
        for (int i = 0; i < featureList.size(); i++) {
            Object[] featureArr = (Object[]) Objects.requireNonNull(
                    EncodeUtils.bytesToObj(Base64.getDecoder().decode(featureList.get(i))));
            for (int j = 0; j < featureArr.length; j++) {
                assertEquals(featureArr[j], itemRowList.get(i).get(j + 1));
            }
        }
    }

    @Test
    public void testGetMetrics() {
        Empty request = Empty.newBuilder().build();

        FeatureProto.ServerMessage msg = null;
        try {
            msg = featureBlockingStub.getMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.error("RPC failed:" + e.getStatus());
        }
        assertNotNull(msg);
        assertNotNull(msg.getStr());
        logger.info("Got metrics: " + msg.getStr());
    }

    @Test
    public void testResetMetrics() {
        Empty request = Empty.newBuilder().build();
        Empty empty = null;
        try {
            empty = featureBlockingStub.resetMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.error("RPC failed: " + e.getStatus().toString());
        }
        assertNotNull(empty);
    }
}
