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

package com.intel.analytics.bigdl.friesian.serving.feature;

import com.google.protobuf.Empty;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.IDs;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.ServerMessage;
import com.intel.analytics.bigdl.friesian.serving.utils.CMDParser;
import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;

import java.util.Base64;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class FeatureClient {
    private static final Logger logger = Logger.getLogger(FeatureClient.class.getName());
    private FeatureGrpc.FeatureBlockingStub blockingStub;

    public FeatureClient(Channel channel) {
        blockingStub = FeatureGrpc.newBlockingStub(channel);
    }

    public Object[] getUserFeatures(int[] userIds) {
        IDs.Builder request = IDs.newBuilder();
        for (int id: userIds) {
            request.addID(id);
        }

        Object[] featureObjs = null;

        Features features;
        try {
            features = blockingStub.getUserFeatures(request.build());
            featureObjs = featuresToObject(features);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus());
        }
        return featureObjs;
    }

    public void getItemFeatures(int[] itemIds) {
        IDs.Builder request = IDs.newBuilder();
        // TODO: insert ids
        for (int id: itemIds) {
            request.addID(id);
        }

        Features features;
        try {
            features = blockingStub.getItemFeatures(request.build());
            Object[] featureObjs = featuresToObject(features);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus());
            return;
        }
    }

    private Object[] featuresToObject(Features features) {
        List<String> b64Features = features.getB64FeatureList();
        Object[] result = new Object[b64Features.size()];
        for (int i = 0; i < b64Features.size(); i ++) {
            if (b64Features.get(i).equals("")) {
                result[i] = null;
            } else {
                byte[] byteBuffer = Base64.getDecoder().decode(b64Features.get(i));
                Object obj = EncodeUtils.bytesToObj(byteBuffer);
                result[i] = obj;
            }
        }
        return result;
    }

    public void getMetrics() {
        Empty request = Empty.newBuilder().build();

        ServerMessage msg;
        try {
            msg = blockingStub.getMetrics(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed:" + e.getStatus());
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
        FeatureClient client = new FeatureClient(channel);

        int userNum = 1000;
        int[] userList = Utils.loadUserData(dir, "enaging_user_id", userNum);
        long start = System.nanoTime();
        for (int userId : userList) {
            Object[] result = client.getUserFeatures(new int[]{userId});
        }
        long end = System.nanoTime();
        double time = (double)(end - start)/(userNum * 1000000);
        System.out.println("Total user number: " + userNum + ", Average search time: " + time
                + " ms");
        client.getMetrics();
        client.resetMetrics();
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
    }
}
