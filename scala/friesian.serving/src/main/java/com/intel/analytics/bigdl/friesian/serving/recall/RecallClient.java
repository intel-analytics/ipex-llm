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

package com.intel.analytics.bigdl.friesian.serving.recall;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage;
import com.intel.analytics.bigdl.friesian.serving.utils.CMDParser;
import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import com.intel.analytics.bigdl.friesian.serving.utils.TimerMetrics;
import com.intel.analytics.bigdl.friesian.serving.utils.TimerMetrics$;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;

import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;


public class RecallClient {
    private static final Logger logger = Logger.getLogger(RecallClient.class.getName());

    private final RecallGrpc.RecallBlockingStub blockingStub;

    private static MetricRegistry metrics = new MetricRegistry();
    private static Timer searchTimer = metrics.timer("recall.search");

    public RecallClient(Channel channel) {
        blockingStub = RecallGrpc.newBlockingStub(channel);
    }

    public void search(int userId, int k) {

        Query request = Query.newBuilder().setUserID(userId).setK(k).build();

        Candidates candidates;
        try {
            candidates = blockingStub.searchCandidates(request);
            printCandidates(userId, candidates);
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

    public void printCandidates(int userId, Candidates candidates) {
        List<Integer> candidateList = candidates.getCandidateList();
        System.out.printf("UserID %d: Candidates: ", userId);
        StringBuilder sb = new StringBuilder();
        for (Integer candidate: candidateList) {
            sb.append(candidate).append('\t');
        }
        sb.append('\n');
        System.out.print(sb);
    }

    /** Issues several different requests and then exits. */
    public static void main(String[] args) throws InterruptedException {
        Logger.getLogger("org").setLevel(Level.ERROR);

        CMDParser cmdParser = new CMDParser();
        cmdParser.addOption("target", "The server to connect to.", "localhost:8980");
        cmdParser.addOption("dataDir", "The data file.", "wnd_user.parquet");

        cmdParser.parseOptions(args);
        String target = cmdParser.getOptionValue("target");
        String dir = cmdParser.getOptionValue("dataDir");

        ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        RecallClient client = new RecallClient(channel);

        int[] userList = Utils.loadUserData(dir, "enaging_user_id", 1000);
        for (int userId: userList) {
            Timer.Context searchContext = searchTimer.time();
            client.search(userId, 50);
            searchContext.stop();
        }
        client.getClientMetrics();
        client.getMetrics();
        client.resetMetrics();
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
    }
}
