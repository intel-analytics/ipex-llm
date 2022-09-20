package com.intel.analytics.bigdl.friesian.serving.recommender;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.dllib.utils.Table;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto;
import com.intel.analytics.bigdl.friesian.serving.utils.TimerMetrics;
import com.intel.analytics.bigdl.friesian.serving.utils.TimerMetrics$;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import com.intel.analytics.bigdl.friesian.serving.utils.recommender.RecommenderUtils;
import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import scala.Tuple2;

import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class RecommenderImpl {

    private static final Logger logger = LogManager.getLogger(RecommenderImpl.class.getName());
    private static RecommenderImpl instance = null;
    private MetricRegistry metrics = new MetricRegistry();
    private final RecallGrpc.RecallBlockingStub recallStub;
    private final FeatureGrpc.FeatureBlockingStub featureStub;
    private final RankingGrpc.RankingBlockingStub rankingStub;
    Timer overallTimer = metrics.timer("recommend.overall");
    Timer recallTimer = metrics.timer("recommend.recall");
    Timer itemFeatureTimer = metrics.timer("recommend.feature.item");
    Timer userFeatureTimer = metrics.timer("recommend.feature.user");
    Timer preprocessTimer = metrics.timer("recommend.preprocess");
    Timer rankingInferenceTimer = metrics.timer("recommend.rankingInference");
    Timer topKTimer = metrics.timer("recommend.topK");

    private RecommenderImpl() {
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
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (recallChannel != null) {
                try {
                    recallChannel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            if (featureChannel != null) {
                try {
                    featureChannel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            if (rankingChannel != null) {
                try {
                    rankingChannel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }));
    }

    public static RecommenderImpl getInstance() {
        if (instance == null) {
            instance = new RecommenderImpl();
        }
        return instance;
    }

    public IDProbList getRecommendIDs(int id, int canK, int k){
        Timer.Context overallContext = overallTimer.time();
        if (canK < k) {
            return new IDProbList(Status.Code.FAILED_PRECONDITION, "CandidateNum" +
                    " should be larger than recommendNum.");
        }

        RecallProto.Candidates candidates;
        try {
            candidates = this.searchCandidates(id, canK);
        } catch (StatusRuntimeException e) {
            e.printStackTrace();
            return new IDProbList(Status.Code.UNAVAILABLE,
                    "recall service unavailable cause: " + e.getCause().toString() +
                            ", message: " + e.getMessage());
        }

        FeatureProto.Features userFeature;
        try {
            userFeature = this.getUserFeature(id);
        } catch (StatusRuntimeException e) {
            e.printStackTrace();
            return new IDProbList(Status.Code.UNAVAILABLE,
                    "feature service unavailable cause: " + e.getCause().toString() +
                    ", message: " + e.getMessage());
        }

        FeatureProto.Features itemFeature;
        try {
            itemFeature = this.getItemFeatures(candidates);
        } catch (StatusRuntimeException e) {
            e.printStackTrace();
            return new IDProbList(Status.Code.UNAVAILABLE,
                    "feature service unavailable cause: " + e.getCause().toString() +
                    ", message: " + e.getMessage());
        }

        Tuple2<int[], Table[]> itemInputTuple;
        try {
            itemInputTuple = this.buildRankingInput(userFeature, itemFeature);
        } catch (Exception e) {
            e.printStackTrace();
            return new IDProbList(Status.Code.FAILED_PRECONDITION, e.getMessage());
        }

        IDProbList idProbList;
        try {
            idProbList = this.inferenceAndRanking(itemInputTuple, k);
        } catch (StatusRuntimeException e) {
            e.printStackTrace();
            return new IDProbList(Status.Code.UNAVAILABLE,
                    "ranking service unavailable cause: " + e.getCause().toString() +
                    ", message: " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            return new IDProbList(Status.Code.UNAVAILABLE, e.getMessage());
        }

        overallContext.stop();

        return idProbList;
    }

    private RecallProto.Candidates searchCandidates(Integer id, int canK) {
        Timer.Context recallContext = recallTimer.time();
        RecallProto.Query query = RecallProto.Query.newBuilder().setUserID(id).setK(canK).build();
        RecallProto.Candidates candidates;
        try {
            candidates = recallStub.searchCandidates(query);
        } catch (StatusRuntimeException e) {
            throw e;
        }
        recallContext.stop();
        return candidates;
    }

    private FeatureProto.Features getUserFeature(Integer id) {
        Timer.Context userFeatureContext = userFeatureTimer.time();
        FeatureProto.IDs userIds = FeatureProto.IDs.newBuilder().addID(id).build();
        FeatureProto.Features userFeature;
        try {
            userFeature = featureStub.getUserFeatures(userIds);
        } catch (StatusRuntimeException e) {
            throw e;
        }
        userFeatureContext.stop();
        return userFeature;
    }

    private FeatureProto.Features getItemFeatures(RecallProto.Candidates candidates) {
        Timer.Context itemFeatureContext = itemFeatureTimer.time();
        FeatureProto.IDs.Builder itemIDsBuilder = FeatureProto.IDs.newBuilder();
        for (Integer itemId: candidates.getCandidateList()) {
            itemIDsBuilder.addID(itemId);
        }
        FeatureProto.IDs itemIDs = itemIDsBuilder.build();
        FeatureProto.Features itemFeature;
        try {
            itemFeature = featureStub.getItemFeatures(itemIDs);
        } catch (StatusRuntimeException e) {
            throw e;
        }
        itemFeatureContext.stop();
        return itemFeature;
    }

    private Tuple2<int[], Table[]> buildRankingInput(FeatureProto.Features userFeature,
                                                     FeatureProto.Features itemFeature) {
        Timer.Context preprocessContext = preprocessTimer.time();
        Tuple2<int[], Table[]> itemInputTuple;
        try {
            itemInputTuple = RecommenderUtils.featuresToRankingInputSet(userFeature,
                    itemFeature, Utils.helper().getInferenceBatch());
        } catch (Exception e) {
            throw e;
        }
        preprocessContext.stop();
        return itemInputTuple;
    }

    private IDProbList inferenceAndRanking(Tuple2<int[], Table[]> itemInputTuple, int k) {
        int[] itemIDArr = itemInputTuple._1;
        Table[] input = itemInputTuple._2;
        Timer.Context rankingContext = rankingInferenceTimer.time();
        String[] result;
        try {
            result = RecommenderUtils.doPredictParallel(input, rankingStub);
        } catch (StatusRuntimeException e) {
            throw e;
        }
        rankingContext.stop();
        Timer.Context topKContext = topKTimer.time();
        Tuple2<int[], float[]> topKIDProbsTuple;
        try {
            topKIDProbsTuple = RecommenderUtils.getTopK(result,
                    itemIDArr, k);
        } catch (Exception e) {
            e.printStackTrace();
            logger.warn("Ranking encounter an error: "+ e.getMessage());
            throw e;
        }
        int[] topKIDs = topKIDProbsTuple._1;
        float[] topKProbs = topKIDProbsTuple._2;
        topKContext.stop();
        return new IDProbList(topKIDs, topKProbs);
    }

    public String getMetrics(List<TimerMetrics> grpcMetrics) {
        JacksonJsonSerializer jacksonJsonSerializer = new JacksonJsonSerializer();
        Set<String> keys = metrics.getTimers().keySet();
        List<TimerMetrics> timerMetrics = keys.stream()
                .map(key ->
                        TimerMetrics$.MODULE$.apply(key, metrics.getTimers().get(key)))
                .collect(Collectors.toList());

        // Add grpcMetrics if grpcMetrics != null
        if (grpcMetrics != null) {
            timerMetrics = Stream.concat(grpcMetrics.stream(), timerMetrics.stream())
                    .collect(Collectors.toList());
        }

        return jacksonJsonSerializer.serialize(timerMetrics);
    }

    public void resetMetrics() {
        metrics = new MetricRegistry();
        overallTimer = metrics.timer("recommend.overall");
        recallTimer = metrics.timer("recommend.recall");
        itemFeatureTimer = metrics.timer("recommend.feature.item");
        userFeatureTimer = metrics.timer("recommend.feature.user");
        preprocessTimer = metrics.timer("recommend.preprocess");
        rankingInferenceTimer = metrics.timer("recommend.rankingInference");
        topKTimer = metrics.timer("recommend.topK");
    }

    public String getClientMetrics() {
        try {
            Empty request = Empty.newBuilder().build();
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
            return sb.toString();
        } catch (Exception e) {
            e.printStackTrace();
            logger.warn("GetClientMetrics encounter an error: "+ e.getMessage());
            return "{'errorMessage': " + e.getCause().toString() + "}";
        }
    }
}
