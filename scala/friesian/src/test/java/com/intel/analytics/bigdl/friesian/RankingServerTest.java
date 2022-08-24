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
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity;
import com.intel.analytics.bigdl.dllib.utils.Table;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingProto;
import com.intel.analytics.bigdl.friesian.serving.ranking.RankingServer;
import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import com.intel.analytics.bigdl.orca.inference.InferenceModel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Base64;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class RankingServerTest {
    private static final Logger logger = LogManager.getLogger(RankingServerTest.class.getName());
    private static RankingGrpc.RankingBlockingStub rankingBlockingStub;
    private static ManagedChannel channel;
    private static InferenceModel inferenceModel;
    private static RankingServer rankingServer;


    /**
     * Sets up the test fixture.
     * (Called before every test case method.)
     */
    @BeforeAll
    public static void setUp() throws Exception {
        String configDir = Objects.requireNonNull(
                RankingServerTest.class.getClassLoader().getResource("testConfig")).getPath();

        rankingServer = new RankingServer(new String[]{"-c", configDir + "/config_ranking_server.yaml"});
        rankingServer.parseConfig();
        rankingServer.build();
        rankingServer.start();

        inferenceModel = Utils.helper().loadInferenceModel(Utils.helper().getModelParallelism(),
                Utils.helper().getModelPath(), Utils.helper().savedModelInputsArr());

        channel = ManagedChannelBuilder.forAddress(
                "localhost", Utils.helper().getServicePort()).usePlaintext().build();
        rankingBlockingStub = RankingGrpc.newBlockingStub(channel);
    }

    /**
     * Tears down the test fixture.
     * (Called after every test case method.)
     */
    @AfterAll
    public static void tearDown() throws InterruptedException {
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        rankingServer.stop();
    }

    @Test
    public void testDoPredict() {
        Table testTable = TestUtils.generateRankingInput(13, 10);
        String encodedStr = Base64.getEncoder().encodeToString(EncodeUtils.objToBytes(testTable));
        RankingProto.Prediction prediction = rankingBlockingStub.doPredict(
                RankingProto.Content.newBuilder().setEncodedStr(encodedStr).build());
        Activity resultActivity = inferenceModel.doPredict(testTable);
        assertEquals(resultActivity.toString(), Objects.requireNonNull(
                EncodeUtils.bytesToObj(Base64.getDecoder().decode(prediction.getPredictStr()))).toString());
    }

    @Test
    public void testGetMetrics() {
        Empty request = Empty.newBuilder().build();

        RankingProto.ServerMessage msg = null;
        try {
            msg = rankingBlockingStub.getMetrics(request);
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
            empty = rankingBlockingStub.resetMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.error("RPC failed: " + e.getStatus().toString());
        }
        assertNotNull(empty);
    }
}
