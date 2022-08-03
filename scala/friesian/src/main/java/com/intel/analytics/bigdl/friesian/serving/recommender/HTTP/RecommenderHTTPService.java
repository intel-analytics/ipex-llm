package com.intel.analytics.bigdl.friesian.serving.recommender.HTTP;

import com.intel.analytics.bigdl.friesian.serving.recommender.IDProbList;
import com.intel.analytics.bigdl.friesian.serving.recommender.RecommenderImpl;
import com.intel.analytics.bigdl.grpc.JacksonJsonSerializer;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;

@Path("recommender")
public class RecommenderHTTPService {
    private final RecommenderImpl impl = RecommenderImpl.getInstance();

    @GET
    @Path("recommend/{id}")
    @Produces(MediaType.APPLICATION_JSON)
    public String getRecommendIDs(@PathParam("id") int id,
                                  @DefaultValue("50") @QueryParam("canK") int canK,
                                  @DefaultValue("10") @QueryParam("k") int k) {
        JacksonJsonSerializer jacksonJsonSerializer = new JacksonJsonSerializer();
        IDProbList recommendList = impl.getRecommendIDs(id, canK, k);
//        System.out.println(recommendList.toString());
        return jacksonJsonSerializer.serialize(recommendList);
    }

    @GET
    @Path("metrics")
    @Produces(MediaType.APPLICATION_JSON)
    public String getMetrics() {
        return impl.getMetrics(null);
    }

    @GET
    @Path("resetMetrics")
    @Produces(MediaType.APPLICATION_JSON)
    public String resetMetrics() {
        impl.resetMetrics();
        return "{success: true}";
    }

    @GET
    @Path("clientMetrics")
    @Produces(MediaType.APPLICATION_JSON)
    public String getClientMetrics() {
        return impl.getClientMetrics();
    }

    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String test(){
        return "Hello";
    }
}
