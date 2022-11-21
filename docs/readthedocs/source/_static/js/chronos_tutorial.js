$(document).ready(function(){
    $("#tutorial details").attr("open",true);
    $('#tutorial details').each(function(){
        $(this).addClass('forecasterSelected');
        $(this).addClass('filterSelected');
    });
    displaySelected();
});

//event when click the tags' buttons
$("details p button").click(function(){
    var id = $(this).val();
    $("#"+id).trigger("click");
});

//func to show a tutorial
function showTutorials(ids){
    ids.forEach(id => {
        $("#"+id).addClass("filterSelected");
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
    //reset all the details 'filterSelected' class
    $('#tutorial details').each(function(){
        $(this).removeClass('filterSelected');
    });

    //get all checked values
    //class checkboxes is specified to avoid selecting toctree checkboxes (arrows)
    var vals = [];

    $('.checkboxes:input:checkbox:checked').each(function (index, item) {
        vals.push($(this).val());
    });

    //reset checkbox and button
    $("#tutorial button").prop("disabled",false);
    $("#tutorial input[type='checkbox']").prop("disabled",false);
    $("#tutorial input[type='checkbox']").parent().css("color","#404040");
    $("#tutorial button").css("color","#404040");

    //show tutorial according to checked values
    if(vals.length==0){
        //choose noting, show all tutorials
        $('#tutorial details').each(function(){
            $(this).addClass('filterSelected');
        });
    }
    //chose something, disable invalid checkboxes and buttons accordingly.
    else if(vals.length==1){
        if(vals.includes("forecast")){
            var ids = ["ChronosForecaster","TuneaForecasting","AutoTS","AutoWIDE",
            "MultvarWIDE","MultstepWIDE","LSTMF","AutoPr","AnomalyDetection",
            "DeepARmodel","TFTmodel","hyperparameter","taxiDataset","distributedFashion",
            "ONNX","Quantize","TCMF","PenalizeUnderestimation",
            "GPUtrainingCPUacceleration","ServeForecaster"];
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
            var ids = ["TuneaForecasting","AutoTS","AutoWIDE","AutoPr",
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
            var ids = ["distributedFashion","TCMF"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","hyperparameter_tuning","onnxruntime","quantization","customized_model"];
            disCheck(disIds);
        }
        else if(vals.includes("customized_model")){
            var ids = ["AutoTS","DeepARmodel","TFTmodel", "GPUtrainingCPUacceleration"];
            ids.forEach(id => {
                var temp=$("#"+id);
            });
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","onnxruntime","quantization","distributed"];
            disCheck(disIds);
        }
    }
    else if(vals.length==2){
        if(vals.includes("forecast") && vals.includes("hyperparameter_tuning")){
            var ids = ["TuneaForecasting","AutoTS","AutoWIDE","AutoPr","hyperparameter","taxiDataset","ONNX"];
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
            var ids = ["DeepARmodel","TFTmodel","AutoTS","GPUtrainingCPUacceleration"];
            showTutorials(ids);
            var disIds = ["anomaly_detection","simulation","onnxruntime","quantization","distributed"];
            disCheck(disIds);
        }
        else if(vals.includes("forecast") && vals.includes("distributed")){
            var ids = ["distributedFashion","TCMF"];
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
            var ids = ["AutoTS"];
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
            var ids = ["AutoTS"];
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

    displaySelected();
});

//event when choosing forecasters
$(".forecasters").click(function(){ 
    //reset all the details 'forecasterSelected' class
    $('#tutorial details').each(function(){
        $(this).removeClass('forecasterSelected');
    });

    //get the selected forecasters
    var forecasters = [];
    $('.forecasters:checked').each(function(){
        forecasters.push($(this).val());
    });

    if(forecasters.length===0){
        //no forecaster is checked, display all
        $('#tutorial details').each(function(){ 
            $(this).addClass('forecasterSelected');
        });
        $(".showingForecaster i").text("All Forecasters");
    }
    else{
        //mark selected forecasters
        $('#tutorial details').each(function(){
            var sons = [];
            $(this).find(".roundbutton").each(function(){
                sons.push($(this).val())
            });

            if(forecasters.every(val => sons.includes(val))){
                $(this).addClass('forecasterSelected');
            }
        });

        let fs = "";
        for(var i=0;i<forecasters.length;i++){
            fs += forecasters[i]+', ';
        }
        fs = fs.substring(0, fs.length - 2);
       
        $(".showingForecaster i").text(fs);
    }

    displaySelected();
});

function displaySelected(){
    $('#tutorial details').each(function(){
        if(($(this).hasClass("forecasterSelected")) && ($(this).hasClass("filterSelected"))){
            $(this).css("display","block");
            $(this).next().css("display","block");
        }
        else{
            $(this).css("display","none");
            $(this).next().css("display","none");
        }
    });
}