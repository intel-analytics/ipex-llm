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

package com.intel.analytics.bigdl.ppml;

import com.intel.analytics.bigdl.grpc.GrpcServerBase;
import com.intel.analytics.bigdl.ppml.psi.PSIServiceImpl;
// import com.intel.analytics.bigdl.ppml.vfl.NNServiceImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * FLServer is BigDL PPML gRPC server used for FL based on GrpcServerBase
 * User could also call main method and parse server type to start gRPC service
 * Supported types: PSI
 */
public class FLServer extends GrpcServerBase {
    private static final Logger logger = LoggerFactory.getLogger(FLServer.class);

    FLServer(String[] args) {
        super(args);
        configPath = "ppml-conf.yaml";
    }
    FLServer() { this(null); }

    @Override
    public void parseConfig() throws IOException {
        FLHelper flHelper = getConfigFromYaml(FLHelper.class, configPath);
        if (flHelper != null) {
            port = flHelper.serverPort;
        }
        // start all services without providing service list
        serverServices.add(new PSIServiceImpl());
//        serverServices.add(new NNServiceImpl());
    }

    public static void main(String[] args) throws Exception {
        FLServer flServer = new FLServer(args);
        flServer.parseConfig();
        // Set aggregator here
        flServer.build();
        flServer.start();
        flServer.blockUntilShutdown();
    }
}
