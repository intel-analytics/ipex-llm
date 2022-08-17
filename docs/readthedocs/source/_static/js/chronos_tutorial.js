//func to show a tutorial
function showTutorials(ids){
    ids.forEach(id => {
        $("#"+id).css("display","block");
        $("#"+id).attr("open","true");
        $("#"+id).next().css("display","block");
    });
}

//func when click the checkboxes
$(".checkboxes").click(function(){
    //get all checked values
    var vals = [];
    $('input:checkbox:checked').each(function (index, item) {
        vals.push($(this).val());
    });

    //reset display
    $("#tutorial details").css("display","none");
    $("#tutorial hr").css("display","none");

    //show tutorial according to checked values
    if(vals.length==0){
        //choose noting, show all tutorials
        $("#tutorial details").css("display","block");
        $("#tutorial details").removeAttr("open");
        $("#tutorial hr").css("display","block");
    }
    if(vals.includes("forecast")){
        var ids = ["ChronosForecaster","TuneaForecasting","AutoTSEstimator","AutoWIDE",
        "MultvarWIDE","MultstepWIDE","LSTMForecaster","AutoProphet","AnomalyDetection",
        "DeepARmodel","TFTmodel","hyperparameter","taxiDataset","distributedFashion",
        "ONNXRuntime","Quantize","TCMFForecaster"];
        showTutorials(ids);
    }
    if(vals.includes("anomaly_detection")){
        var ids = ["DetectAnomaly","Unsupervised","AnomalyDetection"];
        showTutorials(ids);
    }
    if(vals.includes("simulation")){
        var ids = ["SimualateTimeSeriesData"];
        showTutorials(ids);
    }
    if(vals.includes("hyperparameter_tuning")){
        var ids = ["TuneaForecasting","AutoTSEstimator","AutoWIDE","AutoProphet",
        "hyperparameter","taxiDataset","ONNXRuntime"];
        showTutorials(ids);
    }
    if(vals.includes("onnxruntime")){
        var ids = ["ONNXRuntime"];
        showTutorials(ids);
    }
    if(vals.includes("quantization")){
        var ids = ["Quantize"];
        showTutorials(ids);
    }
    if(vals.includes("distributed")){
        var ids = ["distributedFashion","TCMFForecaster"];
        showTutorials(ids);
    }
    if(vals.includes("customized_model")){
        var ids = ["AutoTSEstimator","DeepARmodel","TFTmodel"];
        showTutorials(ids);
    }
});