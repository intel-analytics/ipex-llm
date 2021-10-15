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
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.*;
import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer;
import com.intel.analytics.bigdl.grpc.GrpcServerBase;
import com.intel.analytics.bigdl.orca.inference.InferenceModel;
import com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib.floatArray;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.ServerInterceptors;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import io.prometheus.client.exporter.HTTPServer;
import me.dinowernli.grpc.prometheus.Configuration;
import me.dinowernli.grpc.prometheus.MonitoringServerInterceptor;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import utils.TimerMetrics;
import utils.TimerMetrics$;
import utils.Utils;
import utils.feature.FeatureUtils;
import utils.gRPCHelper;
import utils.recall.RecallUtils;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class RecallServer extends GrpcServerBase {
    private static final Logger logger = Logger.getLogger(RecallServer.class.getName());

    public RecallServer(String[] args) {
        super(args);
        configPath = "config_recall.yaml";
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
                    .intercept(new RecallService().bindService(), monitoringInterceptor));
        } else {
            serverServices.add(new RecallService());
        }
    }

    /**
     * Main method.  This comment makes the linter happy.
     */
    public static void main(String[] args) throws Exception {
        RecallServer indexingServer = new RecallServer(args);
        indexingServer.build();
        if (Utils.runMonitor()) {
            new HTTPServer.Builder()
                    .withPort(Utils.helper().monitorPort()).build();
        }
        indexingServer.start();
        indexingServer.blockUntilShutdown();
    }

    private static class RecallService extends RecallGrpc.RecallImplBase {
        private InferenceModel userModel;
        private InferenceModel itemModel;
        private com.intel.analytics.bigdl.friesian.serving.recall.RecallService indexService;
        private boolean callFeatureService = false;
        private FeatureGrpc.FeatureBlockingStub featureServiceStub;
        MetricRegistry metrics = new MetricRegistry();
        Timer overallTimer = metrics.timer("indexing.overall");
        Timer predictTimer = metrics.timer("indexing.predict");
        Timer faissTimer = metrics.timer("indexing.faiss");

        RecallService() {
            if (Utils.helper().getGetFeatureFromFeatureService()) {
                callFeatureService = true;
                ManagedChannel featureServiceChannel =
                        ManagedChannelBuilder.forTarget(Utils.helper().getFeatureServiceURL())
                                .usePlaintext().build();
                featureServiceStub = FeatureGrpc.newBlockingStub(featureServiceChannel);
            } else {
                userModel = Utils.helper()
                        .loadInferenceModel(Utils.helper().getModelParallelism(),
                                Utils.helper().getUserModelPath(), null);
            }
            // load or build faiss index
            indexService = new com.intel.analytics.bigdl.friesian.serving.recall.RecallService(128);
            if (Utils.helper().loadSavedIndex()) {
                assert(Utils.helper().getIndexPath() != null): "indexPath must be provided " +
                        "if loadSavedIndex=true.";
                indexService.load(Utils.helper().getIndexPath());
            } else {
                if (Utils.helper().getItemModelPath() != null) {
                    itemModel = Utils.helper()
                            .loadInferenceModel(Utils.helper().getModelParallelism(),
                                    Utils.helper().getItemModelPath(), null);
                    Utils.helper().setItemModel(itemModel);
                }
                String dataDir = Utils.helper().getInitialDataPath();
                RecallUtils.loadItemData(indexService, dataDir, itemModel,1000000);
                assert(this.indexService.isTrained());
            }
            System.out.printf("Index service nTotal = %d\n", this.indexService.getNTotal());
        }

        @Override
        public void searchCandidates(Query request,
                                     StreamObserver<Candidates> responseObserver) {
            Candidates candidates;
            try {
                candidates = search(request);
            } catch (Exception e) {
                e.printStackTrace();
                logger.warn(e.getMessage());
                responseObserver.onError(Status.INTERNAL.withDescription(e.getMessage())
                        .asRuntimeException());
                return;
            }
            responseObserver.onNext(candidates);
            responseObserver.onCompleted();
        }

        @Override
        public void addItem(Item request,
                            StreamObserver<Empty> responseObserver) {
            responseObserver.onNext(addItemToIndex(request));
            responseObserver.onCompleted();
        }

        @Override
        public void getMetrics(Empty request,
                               StreamObserver<ServerMessage> responseObserver) {
            responseObserver.onNext(getMetrics());
            responseObserver.onCompleted();
        }

        @Override
        public void resetMetrics(Empty request, StreamObserver<Empty> responseObserver) {
            metrics = new MetricRegistry();
            overallTimer = metrics.timer("indexing.overall");
            predictTimer = metrics.timer("indexing.predict");
            faissTimer = metrics.timer("indexing.faiss");
            responseObserver.onNext(Empty.newBuilder().build());
            responseObserver.onCompleted();
        }

        private Candidates search(Query msg) throws Exception {
            Timer.Context overallContext = overallTimer.time();
            int userId = msg.getUserID();
            int k = msg.getK();
            Timer.Context predictContext = predictTimer.time();
            Activity userFeature;
            if (callFeatureService) {
                IDs userIds = IDs.newBuilder().addID(userId).build();
                Features feature = featureServiceStub.getUserFeatures(userIds);
                Object[] activityList =
                        Arrays.stream(FeatureUtils.featuresToObject(feature))
                                .filter(Objects::nonNull).toArray();
                if (activityList.length == 0) {
                    throw new Exception("Can't get user feature from feature service");
                }
                userFeature = (Activity) activityList[0];
            } else {
                userFeature = this.userModel
                        .doPredict(RecallUtils.constructActivity(Collections.singletonList(userId)));
            }
            predictContext.stop();
            Timer.Context faissContext = faissTimer.time();
            float[] userFeatureList = RecallUtils.activityToFloatArr(userFeature);
            int[] candidates =
                    indexService.search(com.intel.analytics.bigdl.friesian.serving.recall.RecallService.vectorToFloatArray(userFeatureList), k);
            faissContext.stop();
            Candidates.Builder result = Candidates.newBuilder();
            // TODO: length < k
            for (int i = 0; i < k; i ++) {
                result.addCandidate(candidates[i]);
            }
            overallContext.stop();
            return result.build();
        }

        private Empty addItemToIndex(Item msg) {
            // TODO: multi server synchronize
            System.out.printf("Index service nTotal before = %d\n", this.indexService.getNTotal());
            int itemId = msg.getItemID();
            Activity itemFeature = predict(this.itemModel,
                    RecallUtils.constructActivity(Collections.singletonList(itemId)));
            float[] itemFeatureList = RecallUtils.activityToFloatArr(itemFeature);
            addToIndex(itemId, itemFeatureList);
            System.out.printf("Index service nTotal after = %d\n", this.indexService.getNTotal());
            return Empty.newBuilder().build();
        }

        private Activity predict(InferenceModel inferenceModel, Activity data){
            return inferenceModel.doPredict(data);
        }

        private void addToIndex(int targetId, float[] vector) {
            floatArray fa = com.intel.analytics.bigdl.friesian.serving.recall.RecallService.vectorToFloatArray(vector);
            this.indexService.add(targetId, fa);
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
    }
}
