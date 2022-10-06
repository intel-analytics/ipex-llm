package com.intel.analytics.bigdl.friesian.serving.recommender.HTTP;

import com.intel.analytics.bigdl.friesian.serving.recommender.IDProbList;
import com.intel.analytics.bigdl.friesian.serving.recommender.RecommenderImpl;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer;
import io.prometheus.client.Counter;
import io.prometheus.client.Histogram;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;

@Path("recommender")
public class RecommenderHTTPService {
    private final RecommenderImpl impl = RecommenderImpl.getInstance();
    private static final Histogram requestLatency = Histogram.build()
            .namespace("http")
            .name("requests_latency_seconds")
            .help("Request latency in seconds.")
            .buckets(Utils.getPromBuckets())
            .labelNames("http_service", "http_method")
            .register();
    private static final Counter requests = Counter.build()
            .namespace("http")
            .name("requests_total")
            .help("Total requests.")
            .labelNames("http_service", "http_method")
            .register();

    @GET
    @Path("recommend/{id}")
    @Produces(MediaType.APPLICATION_JSON)
    public String getRecommendIDs(@PathParam("id") int id,
                                  @DefaultValue("50") @QueryParam("canK") int canK,
                                  @DefaultValue("10") @QueryParam("k") int k) {
        requests.labels("recommender", "getRecommendIDs").inc();
        Histogram.Timer requestTimer = requestLatency.
                labels("recommender", "getRecommendIDs").startTimer();
        try {
            JacksonJsonSerializer jacksonJsonSerializer = new JacksonJsonSerializer();
            IDProbList recommendList = impl.getRecommendIDs(id, canK, k);
            return jacksonJsonSerializer.serialize(recommendList);
        } finally {
            requestTimer.observeDuration();
        }
    }

    @GET
    @Path("metrics")
    @Produces(MediaType.APPLICATION_JSON)
    public String getMetrics() {
        requests.labels("recommender", "getMetrics").inc();
        return impl.getMetrics(null);
    }

    @GET
    @Path("resetMetrics")
    @Produces(MediaType.APPLICATION_JSON)
    public String resetMetrics() {
        requests.labels("recommender", "resetMetrics").inc();
        impl.resetMetrics();
        return "{success: true}";
    }

    @GET
    @Path("clientMetrics")
    @Produces(MediaType.APPLICATION_JSON)
    public String getClientMetrics() {
        requests.labels("recommender", "getClientMetrics").inc();
        return impl.getClientMetrics();
    }

    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String hello(){
        requests.labels("recommender", "hello").inc();
        return "Hello";
    }
}
