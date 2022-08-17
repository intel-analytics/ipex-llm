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
        String encodedStr = "rO0ABXNyACtjb20uaW50ZWwuYW5hbHl0aWNzLmJpZ2RsLmRsbGliLnV0aWxzLlRhYmxlOO22SWykklICAAJJAAh0b3BJbmRleEwAMmNvbSRpbnRlbCRhbmFseXRpY3MkYmlnZGwkZGxsaWIkdXRpbHMkVGFibGUkJHN0YXRldAAeTHNjYWxhL2NvbGxlY3Rpb24vbXV0YWJsZS9NYXA7eHAAAAANc3IAIHNjYWxhLmNvbGxlY3Rpb24ubXV0YWJsZS5IYXNoTWFwAAAAAAAAAAEDAAB4cHcNAAAC7gAAAA0AAAAEAHNyABFqYXZhLmxhbmcuSW50ZWdlchLioKT3gYc4AgABSQAFdmFsdWV4cgAQamF2YS5sYW5nLk51bWJlcoaslR0LlOCLAgAAeHAAAAAIc3IAMmNvbS5pbnRlbC5hbmFseXRpY3MuYmlnZGwuZGxsaWIudGVuc29yLkRlbnNlVGVuc29yUYzkSde9MaUCAAdJAA5fc3RvcmFnZU9mZnNldEkACm5EaW1lbnNpb25bAAVfc2l6ZXQAAltJTAAIX3N0b3JhZ2V0ADVMY29tL2ludGVsL2FuYWx5dGljcy9iaWdkbC9kbGxpYi90ZW5zb3IvQXJyYXlTdG9yYWdlO1sAB19zdHJpZGVxAH4ACUwANmNvbSRpbnRlbCRhbmFseXRpY3MkYmlnZGwkZGxsaWIkdGVuc29yJERlbnNlVGVuc29yJCRldnQASExjb20vaW50ZWwvYW5hbHl0aWNzL2JpZ2RsL2RsbGliL3RlbnNvci9UZW5zb3JOdW1lcmljTWF0aCRUZW5zb3JOdW1lcmljO0wAPmNvbSRpbnRlbCRhbmFseXRpY3MkYmlnZGwkZGxsaWIkdGVuc29yJERlbnNlVGVuc29yJCRldmlkZW5jZSQxdAAYTHNjYWxhL3JlZmxlY3QvQ2xhc3NUYWc7eHAAAAAAAAAAAXVyAAJbSU26YCZ26rKlAgAAeHAAAAABAAAACnNyADNjb20uaW50ZWwuYW5hbHl0aWNzLmJpZ2RsLmRsbGliLnRlbnNvci5BcnJheVN0b3JhZ2XTuNwZx88lrQIAAkwACmV2aWRlbmNlJDFxAH4ADEwABnZhbHVlc3QAEkxqYXZhL2xhbmcvT2JqZWN0O3hwc3IAJnNjYWxhLnJlZmxlY3QuTWFuaWZlc3RG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";
        RankingProto.Prediction prediction = rankingBlockingStub.doPredict(
                RankingProto.Content.newBuilder().setEncodedStr(encodedStr).build());
        Activity activity = (Activity) EncodeUtils.bytesToObj(Base64.getDecoder().decode(encodedStr));
        Activity resultActivity = inferenceModel.doPredict(activity);
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
