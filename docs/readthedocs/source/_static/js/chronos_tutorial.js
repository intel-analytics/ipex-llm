$(document).ready(function(){
    $("#tutorial details").attr("open",true);
});

//func to show a tutorial
function showTutorials(ids){
    ids.forEach(id => {
        $("#"+id).css("display","block");
        $("#"+id).attr("open","true");
        $("#"+id).next().css("display","block");
    });
}

//func to disable checkbox and button
function disCheck(ids){
    ids.forEach(id => {
        $("#"+id).prop("disabled", true);
        $("#"+id).parent().css("color","#c5c5c5");
        $("button[value='"+id+"']").prop("disabled",true);
        $("button[value='"+id+"']").css("color","#c5c5c5");
    });
}

//event when click the checkboxes
$(".checkboxes").click(function(){
    //get all checked values
    var vals = [];
    $('input:checkbox:checked').each(function (index, item) {
        vals.push($(this).val());
    });

    //reset display
    $("#tutorial details").css("display","none");
    $("#tutorial hr").css("display","none");
    //reset checkbox and button
    $("#tutorial button").prop("disabled",false);
    $("#tutorial input[type='checkbox']").prop("disabled",false);
    $("#tutorial input[type='checkbox']").parent().css("color","#404040");
    $("#tutorial button").css("color","#404040");

    //show tutorial according to checked values
    if(vals.length==0){
        //choose noting, show all tutorials
        $("#tutorial details").css("display","block");
        $("#tutorial details").attr("open",true);
        $("#tutorial hr").css("display","block");
    }
    //chose something, disable invalid checkboxes and buttons accordingly.
    else if(vals.length==1){
        if(vals.includes("forecast")){
            var ids = ["ChronosForecaster","TuneaForecasting","AutoTSEstimator","AutoWIDE",
            "MultvarWIDE","MultstepWIDE","LSTMForecaster","AutoProphet","AnomalyDetection",
            "DeepARmodel","TFTmodel","hyperparameter","taxiDataset","distributedFashion",
            "ONNX","Quantize","TCMFForecaster","PenalizeUnderestimation",
            "GPUtrainingCPUacceleration"];
            showTutorials(ids);
            var disIds = ["simulation"];
            disCheck(disIds);
        }
        else if(vals.includes("anomaly_detection")){
            var ids = ["DetectAnomaly","Unsupervised","AnomalyDetection"];
            showTutorials(ids);
            var disIds = ["simulation","hyperparameter_tuning","onnxruntime","quantization","distributed","customized_model"];
            disCheck(disIds);
        }
        else if(vals.includes("simulation")){
            var ids = ["SimualateTimeSeriesData"];
            showTutorials(ids);
            var disIds = ["forecast","anomaly_detection","hyperparameter_tuning","onnxruntime","quantization","distributed","customized_model"];
            disCheck(disIds);
        }
        else if(vals.includes("hyperparameter_tuning")){
            var ids = ["TuneaForecasting","AutoTSEstimator","AutoWIDE","AutoProphet",
            "hyperparameter","taxiDataset","ONNX"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","quantization","distributed"];
            disCheck(disIds);
        }
        else if(vals.includes("onnxruntime")){
            var ids = ["ONNX"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","quantization","distributed","customized_model"];
            disCheck(disIds);
        }
        else if(vals.includes("quantization")){
            var ids = ["Quantize"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","hyperparameter_tuning","onnxruntime","distributed","customized_model"];
            disCheck(disIds);
        }
        else if(vals.includes("distributed")){
            var ids = ["distributedFashion","TCMFForecaster"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","hyperparameter_tuning","onnxruntime","quantization","customized_model"];
            disCheck(disIds);
        }
        else if(vals.includes("customized_model")){
            var ids = ["AutoTSEstimator","DeepARmodel","TFTmodel", "GPUtrainingCPUacceleration"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","onnxruntime","quantization","distributed"];
            disCheck(disIds);
        }
    }
    else if(vals.length==2){
        if(vals.includes("forecast") && vals.includes("hyperparameter_tuning")){
            var ids = ["TuneaForecasting","AutoTSEstimator","AutoWIDE","AutoProphet","hyperparameter","taxiDataset","ONNX","AutoTSEstimator"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","quantization","distributed"];
            disCheck(disIds);
        }
        else if(vals.includes("forecast") && vals.includes("anomaly_detection")){
            var ids = ["AnomalyDetection"];
            showTutorials(ids);
            var disIds = ["simulation","hyperparameter_tuning","onnxruntime","quantization","distributed","customized_model"];
            disCheck(disIds);
        }
        else if(vals.includes("forecast") && vals.includes("customized_model")){
            var ids = ["DeepARmodel","TFTmodel","AutoTSEstimator","GPUtrainingCPUacceleration"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","onnxruntime","quantization","distributed"];
            disCheck(disIds);
        }
        else if(vals.includes("forecast") && vals.includes("distributed")){
            var ids = ["distributedFashion","TCMFForecaster"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","hyperparameter_tuning","onnxruntime","quantization","customized_model"];
            disCheck(disIds);
        }
        else if(vals.includes("forecast") && vals.includes("quantization")){
            var ids = ["Quantize"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","hyperparameter_tuning","onnxruntime","distributed","customized_model"];
            disCheck(disIds);
        }
        else if(vals.includes("hyperparameter_tuning") && vals.includes("customized_model")){
            var ids = ["AutoTSEstimator"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","onnxruntime","quantization","distributed"];
            disCheck(disIds);
        }
        else if(vals.includes("forecast") && vals.includes("onnxruntime")){
            var ids = ["ONNX"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","quantization","distributed","customized_model"];
            disCheck(disIds);
        }
        else if(vals.includes("hyperparameter_tuning") && vals.includes("onnxruntime")){
            var ids = ["ONNX"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","quantization","distributed","customized_model"];
            disCheck(disIds);
        }
    }
    else if(vals.length==3){
        if(vals.includes("forecast") && vals.includes("hyperparameter_tuning") && vals.includes("customized_model")){
            var ids = ["AutoTSEstimator"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","onnxruntime","quantization","distributed"];
            disCheck(disIds);
        }
        else if(vals.includes("forecast") && vals.includes("hyperparameter_tuning") && vals.includes("onnxruntime")){
            var ids = ["ONNX"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","quantization","distributed","customized_model"];
            disCheck(disIds);
        }
    }
});

//event when click the tags' buttons
$("details p button").click(function(){
    var id = $(this).val();
    $("#"+id).trigger("click");
});

// var allIds = ["forecast","anomaly_detection","simulation","hyperparameter_tuning","onnxruntime","quantization","distributed","customized_model"];