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

import com.codahale.metrics.ConsoleReporter;
import com.codahale.metrics.MetricAttribute;
import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.common.primitives.Floats;
import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Candidates;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Item;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.Query;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recall.RecallProto.ServerMessage;
import com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib.floatArray;
import com.intel.analytics.bigdl.friesian.serving.utils.TimerMetrics;
import com.intel.analytics.bigdl.friesian.serving.utils.TimerMetrics$;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import com.intel.analytics.bigdl.friesian.serving.utils.feature.FeatureUtils;
import com.intel.analytics.bigdl.friesian.serving.utils.gRPCHelper;
import com.intel.analytics.bigdl.friesian.serving.utils.recall.RecallUtils;
import com.intel.analytics.bigdl.grpc.GrpcServerBase;
import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.ServerInterceptors;
import io.grpc.Status;
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

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class RecallServer extends GrpcServerBase {
    private static final Logger logger = LogManager.getLogger(RecallServer.class.getName());

    public RecallServer(String[] args) {
        super(args);
        port = 8084;
        configPath = "config_recall.yaml";
        options.addOption(new Option("p", "port", true,
                "The port to create the server"));
        Configurator.setLevel("org", Level.ERROR);
    }

    /**
     * Main method.  This comment makes the linter happy.
     */
    public static void main(String[] args) throws Exception {
        RecallServer recallServer = new RecallServer(args);
        recallServer.parseConfig();
        recallServer.build();
        if (Utils.runMonitor()) {
            new HTTPServer.Builder()
                    .withPort(Utils.helper().monitorPort()).build();
        }
        recallServer.start();
        recallServer.blockUntilShutdown();
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
                    .intercept(new RecallService().bindService(), monitoringInterceptor));
        } else {
            serverServices.add(new RecallService());
        }
        serverServices.add(new HealthStatusManager().getHealthService());
    }

    private static class RecallService extends RecallGrpc.RecallImplBase {
        MetricRegistry metrics = new MetricRegistry();
        Timer overallTimer = metrics.timer("indexing.overall");
        Timer featureTimer = metrics.timer("indexing.feature");
        Timer decodeTimer = metrics.timer("indexing.decode");
        Timer faissTimer = metrics.timer("indexing.faiss");
        private final IndexService indexService;
        private final FeatureGrpc.FeatureBlockingStub featureServiceStub;

        RecallService() {
            ManagedChannel featureServiceChannel =
                    ManagedChannelBuilder.forTarget(Utils.helper().getFeatureServiceURL())
                            .usePlaintext().build();
            featureServiceStub = FeatureGrpc.newBlockingStub(featureServiceChannel);
            // load faiss index
            indexService = new IndexService(Utils.helper().indexDim());
            assert (Utils.helper().getIndexPath() != null) : "indexPath must be provided";
            indexService.load(Utils.helper().getIndexPath());
            System.out.printf("Index service nTotal = %d\n", this.indexService.getNTotal());
            if (Utils.helper().logInterval() > 0) {
                MetricAttribute[] disAttr = {MetricAttribute.STDDEV, MetricAttribute.P75,
                        MetricAttribute.P999, MetricAttribute.P98, MetricAttribute.M5_RATE,
                        MetricAttribute.M15_RATE};
                ConsoleReporter reporter = ConsoleReporter.forRegistry(metrics)
                        .convertRatesTo(TimeUnit.SECONDS)
                        .convertDurationsTo(TimeUnit.MILLISECONDS)
                        .disabledMetricAttributes(new HashSet<>(Arrays.asList(disAttr)))
                        .build();
                reporter.start(Utils.helper().logInterval(), TimeUnit.MINUTES);
            }
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
            featureTimer = metrics.timer("indexing.feature");
            faissTimer = metrics.timer("indexing.faiss");
            decodeTimer = metrics.timer("indexing.decode");
            responseObserver.onNext(Empty.newBuilder().build());
            responseObserver.onCompleted();
        }

        private Candidates search(Query msg) throws Exception {
            Timer.Context overallContext = overallTimer.time();
            int userId = msg.getUserID();
            int k = msg.getK();
            Timer.Context featureContext = featureTimer.time();
            float[] userFeatureList;
            IDs userIds = IDs.newBuilder().addID(userId).build();
            Features feature = featureServiceStub.getUserFeatures(userIds);
            featureContext.stop();
            Timer.Context decodeContext = decodeTimer.time();
            Object[][] featureList = FeatureUtils.getFeatures(feature);
            if (featureList[0] == null) {
                throw new Exception("Can't get user feature from feature service");
            }
            userFeatureList = RecallUtils.featureObjToFloatArr(featureList[0]);
            decodeContext.stop();
            Timer.Context faissContext = faissTimer.time();
            int[] candidates =
                    indexService.search(IndexService.vectorToFloatArray(userFeatureList), k);
            faissContext.stop();
            Candidates.Builder result = Candidates.newBuilder();
            // TODO: length < k
            for (int i = 0; i < k; i++) {
                result.addCandidate(candidates[i]);
            }
            overallContext.stop();
            return result.build();
        }

        private Empty addItemToIndex(Item msg) {
            // TODO: multi server synchronize
            // TODO: add to index
            System.out.printf("Index service nTotal before = %d\n", this.indexService.getNTotal());
            addToIndex(msg.getItemID(), Floats.toArray(msg.getItemVectorList()));
            System.out.printf("Index service nTotal after = %d\n", this.indexService.getNTotal());
            return Empty.newBuilder().build();
        }

        private void addToIndex(int targetId, float[] vector) {
            floatArray fa = IndexService.vectorToFloatArray(vector);
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
