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
import com.intel.analytics.bigdl.dllib.utils.Table;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto.*;
import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer;
import com.intel.analytics.bigdl.grpc.ZooGrpcServer;
import io.grpc.*;
import io.grpc.stub.StreamObserver;
import io.prometheus.client.exporter.HTTPServer;
import me.dinowernli.grpc.prometheus.Configuration;
import me.dinowernli.grpc.prometheus.MonitoringServerInterceptor;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import scala.Tuple2;
import utils.TimerMetrics;
import utils.TimerMetrics$;
import utils.Utils;
import utils.gRPCHelper;
import utils.recommender.RecommenderUtils;

import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class RecommenderServer extends ZooGrpcServer {
    private static final Logger logger = Logger.getLogger(RecommenderServer.class.getName());

    public RecommenderServer(String[] args) {
        super(args);
        configPath = "config_recommender.yaml";
        Logger.getLogger("org").setLevel(Level.ERROR);
    }

    @Override
    public void parseConfig() throws IOException {
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
                    .intercept(new RecommenderService().bindService(), monitoringInterceptor));
        } else {
            serverServices.add(new RecommenderService());
        }
    }

    /**
     * Main method.  This comment makes the linter happy.
     */
    public static void main(String[] args) throws Exception {
        RecommenderServer recommendServer = new RecommenderServer(args);
        recommendServer.build();
        if (Utils.runMonitor()) {
            new HTTPServer.Builder()
                    .withPort(Utils.helper().monitorPort()).build();
        }
        recommendServer.start();
        recommendServer.blockUntilShutdown();
    }

    private static class RecommenderService extends RecommenderGrpc.RecommenderImplBase {
        private MetricRegistry metrics = new MetricRegistry();
        private RecallGrpc.RecallBlockingStub recallStub;
        private FeatureGrpc.FeatureBlockingStub featureStub;
        private RankingGrpc.RankingBlockingStub rankingStub;
        Timer overallTimer = metrics.timer("recommend.overall");
        Timer recallTimer = metrics.timer("recommend.recall");
        Timer itemFeatureTimer = metrics.timer("recommend.feature.item");
        Timer userFeatureTimer = metrics.timer("recommend.feature.user");
        Timer preprocessTimer = metrics.timer("recommend.preprocess");
        Timer rankingInferenceTimer = metrics.timer("recommend.rankingInference");
        Timer topKTimer = metrics.timer("recommend.topK");

        RecommenderService() {
            ManagedChannel recallChannel =
                    ManagedChannelBuilder.forTarget(Utils.helper().getRecallServiceURL())
                            .usePlaintext().build();
            ManagedChannel featureChannel =
                    ManagedChannelBuilder.forTarget(Utils.helper().getFeatureServiceURL())
                            .usePlaintext().build();
            ManagedChannel rankingChannel =
                    ManagedChannelBuilder.forTarget(Utils.helper().getRankingServiceURL())
                            .usePlaintext().build();
            recallStub = RecallGrpc.newBlockingStub(recallChannel);
            featureStub = FeatureGrpc.newBlockingStub(featureChannel);
            rankingStub = RankingGrpc.newBlockingStub(rankingChannel);
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
                RecallProto.Candidates candidates;
                try {
                    candidates = this.searchCandidates(id, canK);
                } catch (StatusRuntimeException e) {
                    responseObserver.onError(Status.UNAVAILABLE.withDescription("recall " +
                            "service unavailable: " + e.getMessage()).asRuntimeException());
                    return;
                }

                Features userFeature;
                try {
                    userFeature = this.getUserFeature(id);
                } catch (StatusRuntimeException e) {
                    responseObserver.onError(Status.UNAVAILABLE.withDescription("feature " +
                            "service unavailable: " + e.getMessage()).asRuntimeException());
                    return;
                }

                Features itemFeature;
                try {
                    itemFeature = this.getItemFeatures(candidates);
                } catch (StatusRuntimeException e) {
                    e.printStackTrace();
                    logger.warn("FeatureService unavailable: "+ e.getMessage());
                    responseObserver.onError(Status.UNAVAILABLE.withDescription("feature " +
                            "service unavailable: " + e.getMessage()).asRuntimeException());
                    return;
                }

                Tuple2<int[], Table[]> itemInputTuple;
                try {
                     itemInputTuple = this.buildRankingInput(userFeature, itemFeature);
                } catch (Exception e) {
                    e.printStackTrace();
                    logger.warn("FeaturesToRankingInputSet: "+ e.getMessage());
                    responseObserver.onError(Status.FAILED_PRECONDITION
                            .withDescription(e.getMessage()).asRuntimeException());
                    return;
                }

                IDProbs idProb;
                try {
                    idProb = this.inferenceAndRanking(itemInputTuple, k);
                } catch (StatusRuntimeException e) {
                    responseObserver.onError(Status.UNAVAILABLE.withDescription("ranking " +
                            "service unavailable: " + e.getMessage()).asRuntimeException());
                    return;
                }
                resultBuilder.addIDProbList(idProb);
            }
            overallContext.stop();responseObserver.onNext(resultBuilder.build());
            responseObserver.onCompleted();
        }

        @Override
        public void getMetrics(Empty request,
                               StreamObserver<ServerMessage> responseObserver) {
            responseObserver.onNext(getMetrics());
            responseObserver.onCompleted();
        }

        private RecallProto.Candidates searchCandidates(Integer id, int canK) {
            Timer.Context recallContext = recallTimer.time();
            RecallProto.Query query = RecallProto.Query.newBuilder().setUserID(id).setK(canK).build();
            RecallProto.Candidates candidates;
            try {
                candidates = recallStub.searchCandidates(query);
            } catch (StatusRuntimeException e) {
                e.printStackTrace();
                logger.warn("Recall unavailable: "+ e.getMessage());
                throw e;
            }
            recallContext.stop();
            return candidates;
        }

        private Features getUserFeature(Integer id) {
            Timer.Context userFeatureContext = userFeatureTimer.time();
            IDs userIds = IDs.newBuilder().addID(id).build();
            Features userFeature;
            try {
                userFeature = featureStub.getUserFeatures(userIds);
            } catch (StatusRuntimeException e) {
                e.printStackTrace();
                logger.warn("FeatureService unavailable: "+ e.getMessage());
                throw e;
            }
            userFeatureContext.stop();
            return userFeature;
        }

        private Features getItemFeatures(RecallProto.Candidates candidates) {
            Timer.Context itemFeatureContext = itemFeatureTimer.time();
            IDs.Builder itemIDsBuilder = IDs.newBuilder();
            for (Integer itemId: candidates.getCandidateList()) {
                itemIDsBuilder.addID(itemId);
            }
            IDs itemIDs = itemIDsBuilder.build();
            Features itemFeature;
            try {
                itemFeature = featureStub.getItemFeatures(itemIDs);
            } catch (StatusRuntimeException e) {
                e.printStackTrace();
                logger.warn("FeatureService unavailable: "+ e.getMessage());
                throw e;
            }
            itemFeatureContext.stop();
            return itemFeature;
        }

        private Tuple2<int[], Table[]> buildRankingInput(Features userFeature, Features itemFeature) {
            Timer.Context preprocessContext = preprocessTimer.time();
            Tuple2<int[], Table[]> itemInputTuple;
            try {
                itemInputTuple = RecommenderUtils.featuresToRankingInputSet(userFeature,
                        itemFeature, 0);
            } catch (Exception e) {
                e.printStackTrace();
                logger.warn("FeaturesToRankingInputSet: "+ e.getMessage());
                throw e;
            }
            preprocessContext.stop();
            return itemInputTuple;
        }

        private IDProbs inferenceAndRanking(Tuple2<int[], Table[]> itemInputTuple, int k) {
            int[] itemIDArr = itemInputTuple._1;
            Table[] input = itemInputTuple._2;
            Timer.Context rankingContext = rankingInferenceTimer.time();
            String[] result;
            try {
                result = RecommenderUtils.doPredictParallel(input, rankingStub);
            } catch (StatusRuntimeException e) {
                e.printStackTrace();
                logger.warn("Ranking service unavailable: "+ e.getMessage());
                throw e;
            }
            rankingContext.stop();
            Timer.Context topKContext = topKTimer.time();
            Tuple2<int[], float[]> topKIDProbsTuple = RecommenderUtils.getTopK(result,
                    itemIDArr, k);
            int[] topKIDs = topKIDProbsTuple._1;
            float[] topKProbs = topKIDProbsTuple._2;
            IDProbs.Builder idProbBuilder = IDProbs.newBuilder();
            for (int i = 0; i < topKIDs.length; i ++) {
                idProbBuilder.addID(topKIDs[i]);
                idProbBuilder.addProb(topKProbs[i]);
            }
            topKContext.stop();
            return idProbBuilder.build();
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
            overallTimer = metrics.timer("recommend.overall");
            recallTimer = metrics.timer("recommend.recall");
            itemFeatureTimer = metrics.timer("recommend.feature.item");
            userFeatureTimer = metrics.timer("recommend.feature.user");
            preprocessTimer = metrics.timer("recommend.preprocess");
            rankingInferenceTimer = metrics.timer("recommend.rankingInference");
            topKTimer = metrics.timer("recommend.topK");
            responseObserver.onNext(Empty.newBuilder().build());
            responseObserver.onCompleted();
        }

        @Override
        public void getClientMetrics(Empty request,
                                     StreamObserver<ServerMessage> responseObserver) {
            StringBuilder sb = new StringBuilder();
            String vecMetrics = recallStub.getMetrics(request).getStr();
            recallStub.resetMetrics(request);
            sb.append("Recall Service backend metrics:\n");
            sb.append(vecMetrics).append("\n\n");
            String feaMetrics = featureStub.getMetrics(request).getStr();
            featureStub.resetMetrics(request);
            sb.append("Feature Service backend metrics:\n");
            sb.append(feaMetrics).append("\n\n");
            String infMetrics = rankingStub.getMetrics(request).getStr();
            rankingStub.resetMetrics(request);
            sb.append("Inference Service backend metrics:\n");
            sb.append(infMetrics).append("\n\n");
            responseObserver.onNext(ServerMessage.newBuilder().setStr(sb.toString()).build());
            responseObserver.onCompleted();
        }
    }
}
