package com.intel.analytics.bigdl.friesian.serving.recommender;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

@Path("recommender")
public class RecommenderHTTPService {

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public String getIt() {
        return "Got it!";
    }

    @GET
    @Path("{id}")
    @Produces(MediaType.APPLICATION_JSON)
    public String getRecommendIDs(@PathParam("id") int id) {
        return "{'id': " + id + "}";
//        if (match.isPresent()) {
//            return "---Customer---\n" + match.get().toString();
//        } else {
//            return "Customer not found";
//        }
    }
}
