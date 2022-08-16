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

package com.intel.analytics.bigdl.friesian.serving.recommender;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.*;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import com.intel.analytics.bigdl.grpc.GrpcServerBase;
import io.grpc.*;
import io.grpc.services.HealthStatusManager;
import io.grpc.stub.StreamObserver;
import io.prometheus.client.exporter.HTTPServer;
import me.dinowernli.grpc.prometheus.Configuration;
import me.dinowernli.grpc.prometheus.MonitoringServerInterceptor;
import org.apache.commons.cli.Option;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;
import com.intel.analytics.bigdl.friesian.serving.utils.TimerMetrics;
import com.intel.analytics.bigdl.friesian.serving.utils.TimerMetrics$;
import com.intel.analytics.bigdl.friesian.serving.utils.gRPCHelper;

import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class RecommenderServer extends GrpcServerBase {
    private static final Logger logger = LogManager.getLogger(RecommenderServer.class.getName());

    public RecommenderServer(String[] args) {
        super(args);
        port = 8980;
        configPath = "config_recommender.yaml";
        options.addOption(new Option("p", "port", true,
                "The port to create the server"));
        Configurator.setLevel("org", Level.ERROR);
    }

    @Override
    public void parseConfig() throws IOException, InstantiationException, IllegalAccessException {
        Utils.helper_$eq(getConfigFromYaml(gRPCHelper.class, configPath));
        Utils.helper().parseConfigStrings();
        if (Utils.helper() != null && Utils.helper().getServicePort() != -1) {
            port = Utils.helper().getServicePort();
        } else if (cmd.getOptionValue("port") != null) {
            port = Integer.parseInt(cmd.getOptionValue("port"));
        }

        if (Utils.runMonitor()) {
            logger.info("Starting monitoringInterceptor....");
            MonitoringServerInterceptor monitoringInterceptor =
                    MonitoringServerInterceptor.create(Configuration.allMetrics()
                            .withLatencyBuckets(Utils.getPromBuckets()));
            serverDefinitionServices.add(ServerInterceptors
                    .intercept(new RecommenderService().bindService(), monitoringInterceptor));
        } else {
            serverServices.add(new RecommenderService());
        }
        serverServices.add(new HealthStatusManager().getHealthService());
    }

    /**
     * Main method.  This comment makes the linter happy.
     */
    public static void main(String[] args) throws Exception {
        RecommenderServer recommendServer = new RecommenderServer(args);
        recommendServer.parseConfig();
        recommendServer.build();
        if (Utils.runMonitor()) {
            new HTTPServer.Builder()
                    .withPort(Utils.helper().monitorPort()).build();
        }
        recommendServer.start();
        recommendServer.blockUntilShutdown();
    }

    private static class RecommenderService extends RecommenderGrpc.RecommenderImplBase {
        private RecommenderImpl impl;
        private MetricRegistry metrics = new MetricRegistry();
        Timer overallTimer = metrics.timer("recommend.grpc.overall");

        RecommenderService() {
            impl = RecommenderImpl.getInstance();
        }

        @Override
        public void getRecommendIDs(RecommendRequest request,
                                    StreamObserver<RecommendIDProbs> responseObserver) {
            RecommendIDProbs.Builder resultBuilder = RecommendIDProbs.newBuilder();
            Timer.Context overallContext = overallTimer.time();
            List<Integer> ids = request.getIDList();
            int canK = request.getCandidateNum();
            int k = request.getRecommendNum();
            if (canK < k) {
                responseObserver.onError(Status.FAILED_PRECONDITION.withDescription("CandidateNum" +
                        " should be larger than recommendNum.").asRuntimeException());
                return;
            }
            for (Integer id: ids) {
                IDProbList candidates = impl.getRecommendIDs(id, canK, k);
                if (candidates.isSuccess()) {
                    RecommenderProto.IDProbs.Builder idProbBuilder =
                            RecommenderProto.IDProbs.newBuilder();
                    int[] topKIDs = candidates.getIds();
                    float[] topKProbs = candidates.getProbs();
                    for (int i = 0; i < topKIDs.length; i ++) {
                        idProbBuilder.addID(topKIDs[i]);
                        idProbBuilder.addProb(topKProbs[i]);
                    }
                    resultBuilder.addIDProbList(idProbBuilder.build());
                } else{
                    responseObserver.onError(candidates.getgRPCException());
                }
            }
            overallContext.stop();
            responseObserver.onNext(resultBuilder.build());
            responseObserver.onCompleted();
        }

        @Override
        public void getMetrics(Empty request,
                               StreamObserver<ServerMessage> responseObserver) {
            // grpc metrics
            Set<String> keys = metrics.getTimers().keySet();
            List<TimerMetrics> grpcTimerMetrics = keys.stream()
                    .map(key ->
                            TimerMetrics$.MODULE$.apply(key, metrics.getTimers().get(key)))
                    .collect(Collectors.toList());
            String metricsJson = impl.getMetrics(grpcTimerMetrics );
            responseObserver.onNext(RecommenderProto.ServerMessage.newBuilder()
                    .setStr(metricsJson).build());
            responseObserver.onCompleted();
        }

        @Override
        public void resetMetrics(Empty request, StreamObserver<Empty> responseObserver) {
            metrics = new MetricRegistry();
            overallTimer = metrics.timer("recommend.grpc.overall");
            impl.resetMetrics();
            responseObserver.onNext(Empty.newBuilder().build());
            responseObserver.onCompleted();
        }

        @Override
        public void getClientMetrics(Empty request,
                                     StreamObserver<ServerMessage> responseObserver) {
            String metricsStr = impl.getClientMetrics();
            responseObserver.onNext(ServerMessage.newBuilder().setStr(metricsStr).build());
            responseObserver.onCompleted();
        }
    }
}
