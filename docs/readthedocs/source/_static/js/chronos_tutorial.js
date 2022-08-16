//show default tutorial
$("details").css("display","none");
$("hr").css("display","none");
showTutorial("ChronosForecaster")

//func when click the Show-ALl radio
$("#show").click(function(){
    //clear checkboxes
    $(".checkboxes").prop("checked",false);
    //show all the tutorial
    $("details").attr("open","true");
    $("details").css("display","block");
    $("hr").css("display","block");
})

//func to show a tutorial
function showTutorial(id){
    $("#"+id).css("display","block");
    $("#"+id).attr("open","true");
    $("#"+id).next().css("display","block");
}

//func when click the checkboxes
$(".checkboxes").click(function(){
    //clear radio
    $("#show").prop("checked",false);

    //get all checked values
    var vals = [];
    $('input:checkbox:checked').each(function (index, item) {
        vals.push($(this).val());
    });

    //reset display
    $("details").css("display","none");
    $("hr").css("display","none");

    //show tutorial according to checked values
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