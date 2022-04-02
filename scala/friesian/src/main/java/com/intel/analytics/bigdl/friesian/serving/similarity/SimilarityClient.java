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

import com.intel.analytics.bigdl.friesian.serving.grpc.generated.similarity.SimilarityGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.similarity.SimilarityProto.ItemNeighbors;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.similarity.SimilarityProto.IDs;

import com.intel.analytics.bigdl.friesian.serving.utils.CMDParser;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;

import java.util.concurrent.TimeUnit;

public class SimilarityClient {
    private static final Logger logger = LogManager.getLogger(SimilarityClient.class.getName());
    private final SimilarityGrpc.SimilarityBlockingStub blockingStub;

    public SimilarityClient(Channel channel) {
        blockingStub = SimilarityGrpc.newBlockingStub(channel);
    }


    public ItemNeighbors getItemNeighbors(int[] userIds){
       IDs.Builder ids = IDs.newBuilder();
        for (int id: userIds) {
            ids.addID(id);
        }

        ItemNeighbors itemNeighbors = null;
        try{
            itemNeighbors = blockingStub.getItemNeighbors(ids.build());
        }catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
        }
        return itemNeighbors;
    }

    /** Issues several different requests and then exits. */
    public static void main(String[] args) throws InterruptedException {
        Configurator.setLevel("org", Level.ERROR);

        CMDParser cmdParser = new CMDParser();
        cmdParser.addOption("target", "The server to connect to.", "localhost:8980");
        cmdParser.addOption("dataDir", "The data file.", "wnd_user.parquet");
        cmdParser.addOption("k", "The candidate num, default: 50.", "50");

        cmdParser.parseOptions(args);
        String target = cmdParser.getOptionValue("target");
        String dir = cmdParser.getOptionValue("dataDir");
        ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        SimilarityClient client = new SimilarityClient(channel);

        int dataNum = 10;
        int[] userList = Utils.loadItemData(dir, "tweet_id", dataNum);
        ItemNeighbors itemNeighbors = client.getItemNeighbors(userList);

        System.out.println("item:" + itemNeighbors);
        System.out.println("Total user number: " + dataNum);
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
    }
}
