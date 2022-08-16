//func to show a tutorial
function showTutorial(id){
    $("#"+id).css("display","block");
    $("#"+id).attr("open","true");
    $("#"+id).next().css("display","block");
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
    if(vals.includes("A")){
        showTutorial("ChronosForecaster");
    }
    if(vals.includes("B")){
        showTutorial("TuneaForecasting");
    }
    if(vals.includes("C")){
        showTutorial("DetectAnomaly");
        showTutorial("MultvarWIDE");
        showTutorial("LSTMForecaster");
    }
});