package com.intel.analytics.bigdl.friesian.serving.recommender;

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
        System.out.println(recommendList.toString());
        return jacksonJsonSerializer.serialize(recommendList);
    }
}
