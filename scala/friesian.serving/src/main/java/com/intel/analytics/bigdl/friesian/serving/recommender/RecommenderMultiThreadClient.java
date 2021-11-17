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

import com.intel.analytics.bigdl.friesian.serving.grpc.generated.recommender.RecommenderProto;
import com.intel.analytics.bigdl.friesian.serving.utils.CMDParser;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

public class RecommenderMultiThreadClient {
    private static final Logger logger = Logger.getLogger(RecommenderClient.class.getName());

    /** Issues several different requests and then exits. */
    public static void main(String[] args) throws InterruptedException {
        Logger.getLogger("org").setLevel(Level.ERROR);

        CMDParser cmdParser = new CMDParser();
        cmdParser.addOption("target", "The server to connect to.", "localhost:8980");
        cmdParser.addOption("dataDir", "The data file.", "wnd_user.parquet");
        cmdParser.addOption("k", "The candidate num, default: 50.", "50");
        cmdParser.addOption("clientNum", "Concurrent client number.", "1");
        cmdParser.addOption("testNum", "Test case run number.", "1");

        cmdParser.parseOptions(args);
        String target = cmdParser.getOptionValue("target");
        String dir = cmdParser.getOptionValue("dataDir");
        int candidateK = cmdParser.getIntOptionValue("k");
        int concurrentNum = cmdParser.getIntOptionValue("clientNum");
        int testRepeatNum = cmdParser.getIntOptionValue("testNum");

        ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
        RecommenderClient client = new RecommenderClient(channel);
        int dataNum = 1000;
        int[] userList = Utils.loadUserData(dir, "enaging_user_id", dataNum);

        for (int r = 0; r < testRepeatNum; r ++) {
            logger.info("Test round: " + (r + 1));
            ArrayList<RecommenderThread> tList = new ArrayList<>();
            for (int i = 0; i < concurrentNum; i ++) {
                RecommenderThread t = new RecommenderThread(userList, candidateK, channel);
                tList.add(t);
                t.start();
            }
            for (RecommenderThread t: tList) {
                t.join();
            }
            client.getMetrics();
            client.resetMetrics();
            client.getClientMetrics();
            Thread.sleep(10000);
        }
        channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
    }
}

class RecommenderThread extends Thread {
    private int[] userList;
    private int dataNum;
    private int candidateNum;
    private Channel channel;

    RecommenderThread(int[] userList, int candidateNum, Channel channel) {
        this.userList = userList;
        this.dataNum = userList.length;
        this.candidateNum = candidateNum;
        this.channel = channel;
    }

    @Override
    public void run() {
        RecommenderClient client = new RecommenderClient(this.channel);
        long start = System.nanoTime();
        for (int userId: userList){
            RecommenderProto.RecommendIDProbs result = client.getUserRecommends(new int[]{userId}, candidateNum, 10);
        }
        long end = System.nanoTime();
        long time = (end - start)/dataNum;
        System.out.println("Total user number: " + dataNum);
        System.out.println("Average search time: " + time);
    }
}

