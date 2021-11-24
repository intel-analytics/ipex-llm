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

package com.intel.analytics.bigdl.grpc;

import io.grpc.BindableService;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.ServerServiceDefinition;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.ClientAuth;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContext;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
import org.apache.log4j.Logger;

import javax.net.ssl.SSLException;
import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.concurrent.TimeUnit;

/**
 * All Analytics Zoo gRPC servers are based on GrpcServerBase
 * To implement specific gRPC server, overwrite parseConfig() method
 * This class could also be directly used for start a single service
 */
public abstract class GrpcServerBase extends AbstractGrpcBase {
    protected static final Logger logger = Logger.getLogger(GrpcServerBase.class.getName());
    protected int port = 8980;
    protected Server server;
    protected LinkedList<BindableService> serverServices = new LinkedList<BindableService>();
    protected LinkedList<ServerServiceDefinition> serverDefinitionServices =
            new LinkedList<ServerServiceDefinition>();

    // TLS arguments
    String certChainFilePath;
    String privateKeyFilePath;
    String trustCertCollectionFilePath;


    /**
     * One Server could support multiple services.
     * @param args
     */
    public GrpcServerBase(String[] args) {
        this.args = args;
    }

    public void parseConfig() throws Exception {}

    /** Entrypoint of GrpcServerBase */
    public void build() throws Exception {
        parseConfig();
        ServerBuilder builder = ServerBuilder.forPort(port);
        for (BindableService bindableService : serverServices) {
            builder.addService(bindableService);
        }
        for (ServerServiceDefinition serviceDefinition: serverDefinitionServices) {
            builder.addService(serviceDefinition);
        }
        server = builder.maxInboundMessageSize(Integer.MAX_VALUE).build();
    }

    void buildWithTls() throws Exception {
        parseConfig();
        NettyServerBuilder serverBuilder = NettyServerBuilder.forPort(port);
        for (BindableService bindableService : serverServices) {
            serverBuilder.addService(bindableService);
        }
        for (ServerServiceDefinition serviceDefinition: serverDefinitionServices) {
            serverBuilder.addService(serviceDefinition);
        }
        if (certChainFilePath != null && privateKeyFilePath != null) {
            serverBuilder.sslContext(getSslContext());
        }
        server = serverBuilder.build();
    }

    SslContext getSslContext() throws SSLException {
        SslContextBuilder sslClientContextBuilder = SslContextBuilder.forServer(new File(certChainFilePath),
                new File(privateKeyFilePath));
        if (trustCertCollectionFilePath != null) {
            sslClientContextBuilder.trustManager(new File(trustCertCollectionFilePath));
            sslClientContextBuilder.clientAuth(ClientAuth.REQUIRE);
        }
        return GrpcSslContexts.configure(sslClientContextBuilder).build();
    }


    /** Start serving requests. */
    public void start() throws IOException {
        /* The port on which the server should run */
        server.start();
        logger.info("Server started, listening on " + port);
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                try {
                    GrpcServerBase.this.stop();
                } catch (InterruptedException e) {
                    e.printStackTrace(System.err);
                }
                System.err.println("*** server shut down");
            }
        });
    }
    public void stop() throws InterruptedException {
        if (server != null) {
            server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
        }
    }
    /**
     * Await termination on the main thread since the grpc library uses daemon threads.
     */
    public void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }
}
