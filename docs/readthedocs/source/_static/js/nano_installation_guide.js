var inferences=["inferenceyes", "inferenceno"];
var frameworks=["pytorch", "tensorflow"];
var versions=["pytorch_110", "pytorch_111", "pytorch_112", "pytorch_113", "tf2_270"];
var builds=["stable", "nightly"];

var inference="inferenceno";
var framework="pytorch";
var version="pytorch_112";
var build="nightly";

function refresh_cmd(){
    reset_color(frameworks);
    reset_color(inferences);
    reset_color(builds);

    set_color(framework);
    set_color(inference);
    set_color(build);

    var cmd="NA";

    if(framework=="pytorch"){
        $("#version").append("<td colspan='1'>Versions</td>\
        <td colspan='1'><button id='pytorch_113'>torch_113</button>\
        <td colspan='1'><button id='pytorch_112'>torch_112(default)</button>\
        <td colspan='1'><button id='pytorch_111'>torch_111</button>\
        <td colspan='1'><button id='pytorch_110'>torch_110</button></td>");
    }
    else if(framework=="tensorflow"){
        $("#version").append("<td colspan='1'>Versions</td>\
        <td colspan='4'><button id='tf2_270'>tf2_270</button>");
    }
    reset_color(versions);
    set_color(version);

    if(build=="stable"){
        disable(versions);
    }else{
        enable(versions);
    }

        if(framework=="pytorch"){
            if(build=="stable"){
                cmd="pip install bigdl-nano[pytorch]==2.1.0";
            }else if(build=="nightly"){
                if(inference=="inferenceyes"){
                    if(version=="pytorch_110"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch_110,inference]";
                    }else if(version=="pytorch_111"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch_111,inference]";
                    }else if(version=="pytorch_112"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytroch_112,inference]";
                    }else if(version=="pytorch_113"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytroch_113,inference]";
                    }
                    cmd="pip install --pre --upgrade bigdl-nano[inference]";
                }else if(inference="inferenceno"){
                    if(version=="pytorch_110"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch_110]";
                    }else if(version=="pytorch_111"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch_111]";
                    }else if(version=="pytorch_112"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytroch_112]";
                    }else if(version=="pytorch_113"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytroch_113]";
                    }
                    cmd="pip install --pre --upgrade bigdl-nano";
                }
            }
        }else if(framework="tensorflow"){
            if(build="stable"){
                cmd="pip install bigdl-nano[tensorflow]==2.1.0";
            }else if(build=="nightly"){
                if(inference=="inferenceyes"){
                    if (version=="tf2_270"){
                        cmd="pip install --pre --upgrade bigdl-nano[tensorflow,inference]";
                    }
                }else if(inference=="inferenceno"){
                    if(version=="tf2_270"){
                        cmd="pip install --pre --upgrade bigdl-nano[tensorflow]";
                    }
                }
            }
        }
    $("#cmd").html(cmd);
}

function set_color(id){
    $("#"+id).parent().css("background-color","var(--pst-color-primary)");
    $("#"+id).css("color","var(--pst-color-primary-text)");
    $("#"+id).addClass("isset");
}

//reset the color of unselected buttons
function reset_color(list){
    for (btn in list){
        $("#"+list[btn]).parent().css("background-color","transparent");
        $("#"+list[btn]).css("color","var(--pst-color-text-base)");
        $("#"+list[btn]).removeClass("isset");
    }
}

//disable buttons
function disable(list){
    for(btn in list){
        $("#"+list[btn]).css("text-decoration","line-through");
        $("#"+list[btn]).attr("disabled","true");
    }
    reset_color(list);
    for(btn in list){
        $("#"+list[btn]).parent().css("background-color","var(--pst-color-muted)");
    }
}

//enable buttons
function enable(list){
    for(btn in list){
        $("#"+list[btn]).css("text-decoration","none");
        $("#"+list[btn]).attr("disabled",false);
    }
}

//when clicked a button, update variables
$(document).on('click',"button",function(){
    var id = $(this).attr("id");

    if (frameworks.indexOf(id)>=0){
        framework=id;
    }
    else if (builds.indexOf(id)>=0){
        build=id;
    }
    else if (inferences.indexOf(id)>=0){
        inference=id;
    }
    else if (tf2_versions.indexOf(id)>=0){
        tf2_version=id;
    }
    else if (torch_versions.indexOf(id)>=0){
        torch_version=id;
    }

    refresh_cmd();
})

//func to add button hover effect
$(document).on({
    mouseenter: function () {
        if($(this).prop("disabled")!=true){
            $(this).parent().css("background-color","var(--pst-color-primary)");
            $(this).css("color","var(--pst-color-primary-text)");
        }
    },
    mouseleave: function () {
        if(!$(this).hasClass("isset") && $(this).prop("disabled")!=true){
            $(this).parent().css("background-color","transparent");
            $(this).css("color","var(--pst-color-text-base)");
        }
    }
}, "button");

refresh_cmd();
