package com.intel.analytics.bigdl.friesian.serving.recommender.HTTP;

import com.intel.analytics.bigdl.friesian.serving.utils.CMDParser;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import com.intel.analytics.bigdl.friesian.serving.utils.gRPCHelper;
import com.intel.analytics.bigdl.grpc.ConfigParser;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.glassfish.grizzly.http.server.HttpServer;
import org.glassfish.jersey.grizzly2.httpserver.GrizzlyHttpServerFactory;
import org.glassfish.jersey.server.ResourceConfig;

import javax.ws.rs.core.UriBuilder;
import java.io.IOException;
import java.net.URI;

public class RecommenderHTTP {
    private static URI baseUri;
    private static final Logger logger = LogManager.getLogger(RecommenderHTTP.class.getName());
    /**
     * Starts Grizzly HTTP server exposing JAX-RS resources defined in this application.
     * @return Grizzly HTTP server.
     */
    public static HttpServer startServer(String[] args) throws IOException {
        CMDParser cmdParser = new CMDParser();
        cmdParser.addOption("p", "The port to create the server", "8080");
        cmdParser.addOption("c", "config path", "./config_recommender.yaml");

        cmdParser.parseOptions(args);
        String configPath = cmdParser.getOptionValue("c");
        Utils.helper_$eq(ConfigParser.loadConfigFromPath(configPath, gRPCHelper.class));
        Utils.helper().parseConfigStrings();

        int port = cmdParser.getIntOptionValue("p");
        if (Utils.helper() != null && Utils.helper().getServicePort() != -1) {
            port = Utils.helper().getServicePort();
        }

        baseUri = UriBuilder
                .fromUri("http://0.0.0.0/")
                .port(port)
                .build();

        // create a resource config that scans for JAX-RS resources and providers
        // in com.example package
        final ResourceConfig rc = new ResourceConfig()
                .packages("com.intel.analytics.bigdl.friesian.serving.recommender.HTTP");

        // create and start a new instance of grizzly http server
        // exposing the Jersey application at baseUri
        return GrizzlyHttpServerFactory.createHttpServer(baseUri, rc);
    }

    /**
     * Main method.
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        final HttpServer server = startServer(args);

        if (Utils.runMonitor()) {
            logger.info("Starting prometheus client HTTPServer at port " +
                    Utils.helper().monitorPort() + "....");
            final io.prometheus.client.exporter.HTTPServer monitorServer =
                    new io.prometheus.client.exporter.HTTPServer.Builder()
                            .withPort(Utils.helper().monitorPort())
                            .build();
            // register shutdown hook
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                logger.info("Stopping prometheus client HTTPServer....");
                monitorServer.close();
            }, "prometheusShutdownHook"));
        }

        // register shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            logger.info("Stopping server..");
            server.shutdown();
        }, "shutdownHook"));

        // run
        try {
            server.start();
            logger.info(String.format("Recommender Jersey app started with endpoints " +
                    "available at %s%nHit Ctrl-C to stop it...", baseUri.toString()));
            Thread.currentThread().join();
        } catch (Exception e) {
            logger.error("There was an error while starting Grizzly HTTP server.", e);
        }
    }
}
