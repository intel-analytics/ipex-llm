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

package com.intel.analytics.bigdl.friesian.serving.similarity;

import com.intel.analytics.bigdl.friesian.serving.feature.utils.LettuceUtils;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.similarity.SimilarityGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.similarity.SimilarityProto;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.similarity.SimilarityProto.ItemNeighbors;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import com.intel.analytics.bigdl.friesian.serving.utils.gRPCHelper;
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

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class SimilarityServer extends GrpcServerBase {
    private static final Logger logger = LogManager.getLogger(SimilarityServer.class.getName());

    public SimilarityServer(String[] args) {
        super(args);
        port = 8085;
        configPath = "config_similarity.yaml";
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
                    .intercept(new SimilarityService().bindService(), monitoringInterceptor));
        } else {
            serverServices.add(new SimilarityService());
        }
        serverServices.add(new HealthStatusManager().getHealthService());
    }

    /**
     * Main method.  This comment makes the linter happy.
     */
    public static void main(String[] args) throws Exception {
        SimilarityServer smilarityServer = new SimilarityServer(args);
        smilarityServer.build();
        if (Utils.runMonitor()) {
            new HTTPServer.Builder()
                    .withPort(Utils.helper().monitorPort()).build();
        }
        smilarityServer.start();
        smilarityServer.blockUntilShutdown();
    }

    private static class SimilarityService extends SimilarityGrpc.SimilarityImplBase {
        private LettuceUtils redis;

        SimilarityService() {
            redis = LettuceUtils.getInstance(Utils.helper().redisTypeEnum(), Utils.helper().redisHostPort(),
                    Utils.helper().getRedisKeyPrefix(), Utils.helper().redisSentinelMasterURL(),
                    Utils.helper().redisSentinelMasterName(), Utils.helper().itemSlotType());
        }

        @Override
        public void getItemNeighbors(SimilarityProto.IDs request, StreamObserver<ItemNeighbors> responseObserver) {
            List<Integer> ids = request.getIDList();
            ItemNeighbors neighbors;
            int k = 5;
            for (Integer id : ids) {
                try {
                    neighbors = this.getNeighborsFromRedis(id, k);
                } catch (StatusRuntimeException e) {
                    responseObserver.onError(Status.UNAVAILABLE.withDescription("similarity " +
                            "service unavailable: " + e.getMessage()).asRuntimeException());
                    return;
                }
                responseObserver.onNext(neighbors);
                responseObserver.onCompleted();
            }
        }

        private ItemNeighbors getNeighborsFromRedis(Integer id, int k) {

            List<Integer> ids = Collections.singletonList(id);
            String keyPrefix = "neighbor_";

            ItemNeighbors.Builder neighborBuilder = ItemNeighbors.newBuilder();
            List<String> values = redis.MGet(keyPrefix, ids);
            neighborBuilder.addAllID(ids);
            neighborBuilder.addAllSimilarItems(values);

            return neighborBuilder.build();

        }

    }
}
