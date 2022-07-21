package com.intel.analytics.bigdl.friesian.serving.recommender.HTTP;

import com.intel.analytics.bigdl.friesian.serving.utils.CMDParser;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import com.intel.analytics.bigdl.friesian.serving.utils.gRPCHelper;
import com.intel.analytics.bigdl.grpc.ConfigParser;
import org.glassfish.grizzly.http.server.HttpServer;
import org.glassfish.jersey.grizzly2.httpserver.GrizzlyHttpServerFactory;
import org.glassfish.jersey.server.ResourceConfig;

import javax.ws.rs.core.UriBuilder;
import java.io.IOException;
import java.net.URI;

public class RecommenderHTTP {
    private static URI baseUri;
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

        baseUri = UriBuilder
                .fromUri("http://0.0.0.0/")
                .port(cmdParser.getIntOptionValue("p"))
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
        System.out.println(String.format("Recommender Jersey app started with endpoints " +
                "available at %s%nHit Ctrl-C to stop it...", baseUri.toString()));
        System.in.read();
        server.shutdownNow();
    }
}
