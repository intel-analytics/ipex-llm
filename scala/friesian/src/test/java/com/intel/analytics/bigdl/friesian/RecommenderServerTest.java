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
import com.intel.analytics.bigdl.friesian.nearline.recall.RecallInitializer;
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils;
import com.intel.analytics.bigdl.friesian.serving.feature.FeatureServer;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto;
import com.intel.analytics.bigdl.friesian.serving.ranking.RankingServer;
import com.intel.analytics.bigdl.friesian.serving.recall.RecallServer;
import com.intel.analytics.bigdl.friesian.serving.recommender.RecommenderServer;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Objects;
import java.util.concurrent.TimeUnit;

import static com.intel.analytics.bigdl.friesian.JavaTestUtils.destroyLettuceUtilsInstance;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class RecommenderServerTest {
    private static final Logger logger = LogManager.getLogger(RecommenderServerTest.class.getName());
    private static RecommenderGrpc.RecommenderBlockingStub recommenderBlockingStub;
    private static ManagedChannel channel;
    private static Row userDataRow;
    private static FeatureServer featureVecServer;
    private static FeatureServer featureServer;
    private static RecallServer recallServer;
    private static RankingServer rankingServer;
    private static RecommenderServer recommenderServer;


    /**
     * Sets up the test fixture.
     * (Called before every test case method.)
     */
    @BeforeAll
    public static void setUp() throws Exception {

        String configDir = Objects.requireNonNull(
                RecommenderServerTest.class.getClassLoader().getResource("testConfig")).getPath();

        FeatureInitializer.main(new String[]{"-c", configDir + "/config_feature_init.yaml"});
        destroyLettuceUtilsInstance();

        FeatureInitializer.main(new String[]{"-c", configDir + "/config_feature_vec_init.yaml"});
        SparkSession sparkSession = SparkSession.builder().getOrCreate();
        userDataRow = sparkSession.read().parquet(NearlineUtils.helper().getInitialUserDataPath()).first();
        destroyLettuceUtilsInstance();

        RecallInitializer.main(new String[]{"-c", configDir + "/config_recall_init.yaml"});

        featureVecServer = new FeatureServer(
                new String[]{"-c", configDir + "/config_feature_vec_server.yaml"});
        featureVecServer.parseConfig();
        featureVecServer.build();
        featureVecServer.start();
        destroyLettuceUtilsInstance();

        featureServer = new FeatureServer(
                new String[]{"-c", configDir
                        + "/config_feature_server.yaml"});
        featureServer.parseConfig();
        featureServer.build();
        featureServer.start();
        destroyLettuceUtilsInstance();

        recallServer = new RecallServer(new String[]{"-c", configDir + "/config_recall_server.yaml"});
        recallServer.parseConfig();
        recallServer.build();
        recallServer.start();

        rankingServer = new RankingServer(new String[]{"-c", configDir + "/config_ranking_server.yaml"});
        rankingServer.parseConfig();
        rankingServer.build();
        rankingServer.start();

        recommenderServer = new RecommenderServer(
                new String[]{"-c", configDir + "/config_recommender_server.yaml"});
        recommenderServer.parseConfig();
        recommenderServer.build();
        recommenderServer.start();

        channel = ManagedChannelBuilder.forAddress(
                "localhost", Utils.helper().getServicePort()).usePlaintext().build();
        recommenderBlockingStub = RecommenderGrpc.newBlockingStub(channel);
    }

    /**
     * Tears down the test fixture.
     * (Called after every test case method.)
     */
    @AfterAll
    public static void tearDown() throws InterruptedException {
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        featureServer.stop();
        featureVecServer.stop();
        recallServer.stop();
        rankingServer.stop();
        recommenderServer.stop();
    }

    @Test
    public void testGetRecommendIDs() {
        RecommenderProto.RecommendIDProbs result = recommenderBlockingStub.getRecommendIDs(
                RecommenderProto.RecommendRequest
                        .newBuilder().addID(userDataRow.getInt(0)).setRecommendNum(3).setCandidateNum(10).build());
        assertEquals(result.getIDProbList(0).getIDCount(), 3);
        logger.info("Got Recommend IDs" + result.getIDProbListList().toString());
    }


    @Test
    public void testGetMetrics() {
        Empty request = Empty.newBuilder().build();

        RecommenderProto.ServerMessage msg = null;
        try {
            msg = recommenderBlockingStub.getMetrics(request);
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
            empty = recommenderBlockingStub.resetMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.error("RPC failed: " + e.getStatus().toString());
        }
        assertNotNull(empty);
    }

    @Test
    public void testGetClientMetrics() {
        Empty request = Empty.newBuilder().build();

        RecommenderProto.ServerMessage msg = null;
        try {
            msg = recommenderBlockingStub.getClientMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.error("RPC failed:" + e.getStatus());
        }
        assertNotNull(msg);
        assertNotNull(msg.getStr());
        logger.info("Got client metrics: " + msg.getStr());
    }
}
