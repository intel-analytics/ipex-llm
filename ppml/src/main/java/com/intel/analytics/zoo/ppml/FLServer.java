/*
 * Copyright 2021 The Analytic Zoo Authors
 *
 * Licensed under the Apache License,  Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,  software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.ppml;

import com.intel.analytics.zoo.grpc.ZooGrpcServer;
import com.intel.analytics.zoo.ppml.psi.PSIServiceImpl;
import io.grpc.BindableService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * FLServer is Analytics Zoo PPML gRPC server used for FL based on ZooGrpcServer
 * User could also call main method and parse server type to start gRPC service
 * Supported types: PSI
 */
public class FLServer extends ZooGrpcServer {
    private static final Logger logger = LoggerFactory.getLogger(FLServer.class);

    FLServer(String[] args, BindableService service) {
        super(args, service);
        configPath = "ppml-conf.yaml";
    }
    FLServer(String[] args) {
        this(args, null);
    }

    @Override
    public void parseConfig() throws IOException {
        FLHelper flHelper = getConfigFromYaml(FLHelper.class, configPath);
        if (flHelper != null) {
            serviceList = flHelper.servicesList;
            port = flHelper.serverPort;
        }
        for (String service : serviceList.split(",")) {
            if (service.equals("psi")) {
                serverServices.add(new PSIServiceImpl());
            } else if (service.equals("ps")) {
                // TODO: add algorithms here
            } else {
                logger.warn("Type is not supported, skipped. Type: " + service);
            }
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        FLServer flServer = new FLServer(args);
        flServer.build();
        flServer.start();
        flServer.blockUntilShutdown();
    }
}
