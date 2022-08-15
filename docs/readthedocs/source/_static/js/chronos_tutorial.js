$(".radios").click(function(){
    let choice = $("input[name=choice]:checked").val();
    if(choice=="A"){
        $("details").removeAttr('open');
        $("#ChronosForecaster").attr("open","true");
    }
    else if(choice=="B"){
        $("details").removeAttr('open');
        $("#TuneaForecasting").attr("open","true");
    }
    else if(choice=="C"){
        $("details").removeAttr('open');
        $("#DetectAnomaly").attr("open","true");
        $("#MultvarWIDE").attr("open","true");
        $("#LSTMForecaster").attr("open","true");
    }
    else if(choice=="open"){
        $("details").attr("open","true");
    }
    else if(choice=="close"){
        $("details").removeAttr('open');
    }
})