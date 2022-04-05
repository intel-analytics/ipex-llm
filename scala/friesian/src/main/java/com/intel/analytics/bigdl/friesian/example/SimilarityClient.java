/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.friesian.example;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage;
import com.intel.analytics.bigdl.friesian.serving.utils.CMDParser;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import com.intel.analytics.bigdl.friesian.serving.utils.TimerMetrics;
import com.intel.analytics.bigdl.friesian.serving.utils.TimerMetrics$;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;

import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class SimilarityClient {
    private static final Logger logger = LogManager.getLogger(com.intel.analytics.bigdl.friesian.example.SimilarityClient.class.getName());

    private final RecallGrpc.RecallBlockingStub blockingStub;

    private static MetricRegistry metrics = new MetricRegistry();
    private static Timer searchTimer = metrics.timer("friesian.similarity");

    public SimilarityClient(Channel channel) {
        blockingStub = RecallGrpc.newBlockingStub(channel);
    }

    public void search(int id, int k) {

        Query request = Query.newBuilder().setUserID(id).setK(k).build();

        Candidates candidates;
        try {
            candidates = blockingStub.searchCandidates(request);
            printCandidates(id, candidates);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
        }
    }

    public void getMetrics() {
        Empty request = Empty.newBuilder().build();

        ServerMessage msg;
        try {
            msg = blockingStub.getMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            return;
        }

        logger.info("Got metrics: " + msg.getStr());
    }

    public void resetMetrics() {
        Empty request = Empty.newBuilder().build();
        try {
            blockingStub.resetMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
        }
    }

    public void getClientMetrics() {
        JacksonJsonSerializer jacksonJsonSerializer = new JacksonJsonSerializer();
        Set<String> keys = metrics.getTimers().keySet();
        List<TimerMetrics> timerMetrics = keys.stream()
                .map(key ->
                        TimerMetrics$.MODULE$.apply(key, metrics.getTimers().get(key)))
                .collect(Collectors.toList());
        String jsonStr = jacksonJsonSerializer.serialize(timerMetrics);
        logger.info("Client metrics: " + jsonStr);
    }

    public void resetClientMetrics() {
        metrics = new MetricRegistry();
        searchTimer = metrics.timer("recall.search");
    }

    public void printCandidates(int id, Candidates candidates) {
        List<Integer> candidateList = candidates.getCandidateList();
        System.out.printf("ID %d: Candidates: ", id);
        StringBuilder sb = new StringBuilder();
        for (Integer candidate: candidateList) {
            sb.append(candidate).append('\t');
        }
        sb.append('\n');
        System.out.print(sb);
    }

    /** Issues several different requests and then exits. */
    public static void main(String[] args) throws InterruptedException {
        Configurator.setLevel("org", Level.ERROR);

        CMDParser cmdParser = new CMDParser();
        cmdParser.addOption("target", "The server to connect to.", "localhost:8085");
        cmdParser.addOption("dataDir", "The data file.", "item_ebd.parquet");

        cmdParser.parseOptions(args);
        String target = cmdParser.getOptionValue("target");
        String dir = cmdParser.getOptionValue("dataDir");

        ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        com.intel.analytics.bigdl.friesian.example.SimilarityClient client = new com.intel.analytics.bigdl.friesian.example.SimilarityClient(channel);

        int[] item_list = Utils.loadUserData(dir, "tweet_id", 100);

        for (int id: item_list) {
            Timer.Context searchContext = searchTimer.time();
            client.search(id, 10);
            searchContext.stop();
        }
        client.getClientMetrics();
        client.getMetrics();
        client.resetMetrics();
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
    }
}
