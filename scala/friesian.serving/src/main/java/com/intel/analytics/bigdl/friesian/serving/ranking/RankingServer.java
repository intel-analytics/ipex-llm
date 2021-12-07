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

package com.intel.analytics.bigdl.friesian.serving.ranking;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingProto.*;
import com.intel.analytics.bigdl.friesian.serving.utils.*;
import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer;
import com.intel.analytics.bigdl.grpc.GrpcServerBase;
import com.intel.analytics.bigdl.orca.inference.InferenceModel;
import io.grpc.ServerInterceptors;
import io.grpc.stub.StreamObserver;
import io.prometheus.client.exporter.HTTPServer;
import me.dinowernli.grpc.prometheus.Configuration;
import me.dinowernli.grpc.prometheus.MonitoringServerInterceptor;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.Base64;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class RankingServer extends GrpcServerBase {
    private static final Logger logger = Logger.getLogger(RankingServer.class.getName());

    /**
     * One Server could support multiple services.
     *
     * @param args
     */
    public RankingServer(String[] args) {
        super(args);
        configPath = "config_ranking.yaml";
        Logger.getLogger("org").setLevel(Level.ERROR);
    }

    @Override
    public void parseConfig() throws IOException, InstantiationException, IllegalAccessException {
        Utils.helper_$eq(getConfigFromYaml(gRPCHelper.class, configPath));
        Utils.helper().parseConfigStrings();
        if (Utils.helper() != null) {
            port = Utils.helper().getServicePort();
        }

        if (Utils.runMonitor()) {
            logger.info("Starting monitoringInterceptor....");
            MonitoringServerInterceptor monitoringInterceptor =
                    MonitoringServerInterceptor.create(Configuration.allMetrics()
                            .withLatencyBuckets(Utils.getPromBuckets()));
            serverDefinitionServices.add(ServerInterceptors
                    .intercept(new RankingService().bindService(), monitoringInterceptor));
        } else {
            serverServices.add(new RankingService());
        }
    }

    /**
     * Main method.  This comment makes the linter happy.
     */
    public static void main(String[] args) throws Exception {
        RankingServer rankingServer = new RankingServer(args);
        rankingServer.build();
        if (Utils.runMonitor()) {
            new HTTPServer.Builder()
                    .withPort(Utils.helper().monitorPort()).build();
        }
        rankingServer.start();
        rankingServer.blockUntilShutdown();
    }

    private static class RankingService extends RankingGrpc.RankingImplBase {
        private final InferenceModel model;
        private MetricRegistry metrics = new MetricRegistry();
        Timer overallTimer = metrics.timer("ranking.overall");
        Timer decodeTimer = metrics.timer("ranking.decode");
        Timer inferenceTimer = metrics.timer("ranking.inference");
        Timer encodeTimer = metrics.timer("ranking.encode");

        RankingService() {
            gRPCHelper helper = Utils.helper();
            this.model = helper.loadInferenceModel(helper.modelParallelism(), helper.modelPath(),
                    helper.savedModelInputsArr());
        }

        @Override
        public void doPredict(Content request,
                              StreamObserver<Prediction> responseObserver) {
            responseObserver.onNext(predict(request));
            responseObserver.onCompleted();
        }

        private Prediction predict(Content msg) {
            Timer.Context overallContext = overallTimer.time();
            String encodedStr = msg.getEncodedStr();
            Timer.Context decodeContext = decodeTimer.time();
            byte[] bytes1 = Base64.getDecoder().decode(encodedStr);
            Activity input = (Activity) EncodeUtils.bytesToObj(bytes1);
            decodeContext.stop();
            Timer.Context inferenceContext = inferenceTimer.time();
            Activity predictResult = model.doPredict(input);
            inferenceContext.stop();
            Timer.Context encodeContext = encodeTimer.time();
            String res = Base64.getEncoder().encodeToString(EncodeUtils.objToBytes(predictResult));
            encodeContext.stop();
            overallContext.stop();
            return Prediction.newBuilder().setPredictStr(res).build();
        }

        @Override
        public void getMetrics(Empty request,
                               StreamObserver<ServerMessage> responseObserver) {
            responseObserver.onNext(getMetrics());
            responseObserver.onCompleted();
        }

        private ServerMessage getMetrics() {
            JacksonJsonSerializer jacksonJsonSerializer = new JacksonJsonSerializer();
            Set<String> keys = metrics.getTimers().keySet();
            List<TimerMetrics> timerMetrics = keys.stream()
                    .map(key ->
                            TimerMetrics$.MODULE$.apply(key, metrics.getTimers().get(key)))
                    .collect(Collectors.toList());
            String jsonStr = jacksonJsonSerializer.serialize(timerMetrics);
            return ServerMessage.newBuilder().setStr(jsonStr).build();
        }

        @Override
        public void resetMetrics(Empty request, StreamObserver<Empty> responseObserver) {
            metrics = new MetricRegistry();
            overallTimer = metrics.timer("ranking.overall");
            decodeTimer = metrics.timer("ranking.decode");
            inferenceTimer = metrics.timer("ranking.inference");
            encodeTimer = metrics.timer("ranking.encode");
            responseObserver.onNext(Empty.newBuilder().build());
            responseObserver.onCompleted();
        }
    }
}
