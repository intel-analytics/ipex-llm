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

package com.intel.analytics.bigdl.ppml.psi.test;

import com.intel.analytics.bigdl.ppml.FLClient;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class NetworkCheckClient{
    public static void main(String[] args) throws Exception {
        String taskID;
        String target;
        // Number of arguments to be passed.
        int argNum = 3;
        if (args.length == 0) {
            //logger.info("No argument passed, using default parameters.");
            taskID = "taskID";
            target = "localhost:50051";
        } else if (args.length < argNum || args.length > argNum + 1) {
            //logger.info("Error: detecting " + Integer.toString(args.length) + " arguments. Expecting " + Integer.toString(argNum) + ".");
            //logger.info("Usage: BenchmarkClient taskID ServerIP ServerPort");
            taskID = "";
            target = "";
            System.exit(0);
        } else {
            taskID = args[0];
            target = args[1] + ":" + args[2];
        }
        //logger.info("TaskID is: " + taskID);
        //logger.info("Accessing service at: " + target);

        int max_wait = 2000;
        // Example code for flClient
        int idSize = 150000;
        // Quick lookup for the plaintext of hashed ids
        HashMap<String, String> data = TestUtils.getRandomHashSetOfStringForFiveFixed(idSize);//Utils.genRandomHashSet(idSize);
        HashMap<String, String> hashedIds = new HashMap<>();
        List<String> hashedIdArray;
        String salt;
        List<String> ids = new ArrayList<>(data.keySet());

        // Create a communication channel to the server,  known as a Channel. Channels are thread-safe
        // and reusable. It is common to create channels at the beginning of your application and reuse
        // them until the application shuts down.
        ManagedChannel channel = ManagedChannelBuilder.forTarget(target)
                // Channels are secure by default (via SSL/TLS).
                //extend message size of server to 200M to avoid size conflict
		.maxInboundMessageSize(209715200)
		.usePlaintext()
                .build();
        try {
            String[] arg = {"-c", BenchmarkClient.class.getClassLoader()
                    .getResource("psi/psi-conf.yaml").getPath()};
            FLClient flClient = new FLClient(arg);
            flClient.build();
            // Get salt from Server
            salt = flClient.getSalt();
            //logger.debug("Client get Slat=" + salt);
            // Hash(IDs, salt) into hashed IDs
            hashedIdArray = TestUtils.parallelToSHAHexString(ids, salt);
            for (int i = 0; i < ids.size(); i++) {
                hashedIds.put(hashedIdArray.get(i), ids.get(i));
            }
            //logger.debug("HashedIDs Size = " + hashedIds.size());
            flClient.uploadSet(hashedIdArray);
            List<String> intersection;

            while (max_wait > 0) {
                intersection = flClient.downloadIntersection();
                if (intersection == null) {
                    //logger.info("Wait 1000ms");
                    Thread.sleep(1000);
                } else {
                    System.out.println("Intersection successful. Intersection's size is " + intersection.size() + ".");
                    break;
                }
                max_wait--;
            }

        } finally {
            // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
            // resources the channel should be shut down when it will no longer be used. If it may be used
            // again leave it running.
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}


