var inferences=["inferenceyes", "inferenceno"];
var frameworks=["pytorch", "tensorflow"];
var versions=["pytorch_110", "pytorch_111", "pytorch_112", "pytorch_113", "tf2_27", "tf2_28", "tf2_29", "tf2_210"];
var releases=["stable", "nightly"];

var inference="inferenceyes";
var framework="pytorch";
var version="pytorch_113";
var release="nightly";

function refresh_cmd(){
    reset_color(frameworks);
    reset_color(inferences);
    reset_color(releases);

    set_color(framework);
    set_color(inference);
    set_color(release);

    var cmd="NA";

    $("#version").empty();
    if(framework=="pytorch"){
        $("#version").append("<td colspan='1'>Pytorch Version</td>\
        <td colspan='1'><button id='pytorch_113'>1.13</button></td>\
        <td colspan='1'><button id='pytorch_112'>1.12</button></td>\
        <td colspan='1'><button id='pytorch_111'>1.11</button></td>\
        <td colspan='1'><button id='pytorch_110'>1.10</button></td>");
    }
    else if(framework=="tensorflow"){
        $("#version").append("<td colspan='1'>Tensorflow Version</td>\
        <td colspan='1'><button id='tf2_210'>2.10</button></td>\
        <td colspan='1'><button id='tf2_29'>2.9</button></td>\
        <td colspan='1'><button id='tf2_28'>2.8</button></td>\
        <td colspan='1'><button id='tf2_27'>2.7</button></td>");
    }
    reset_color(versions);
    set_color(version);

    // if (framework=="pytorch"){
    //     document.getElementById("cmd").style.whiteSpace = "normal";
    // }
    // else{
    //     document.getElementById('cmd').style.whiteSpace = "nowrap";
    // }
        if(framework=="pytorch"){
            if(release=="stable"){
                if(inference=="inferenceyes"){
                    if(version=="pytorch_110"){
                        cmd="pip install bigdl-nano[pytorch_110,inference] -f https://software.intel.com/ipex-whl-stable";
                    }else if(version=="pytorch_111"){
                        cmd="pip install bigdl-nano[pytorch_111,inference]";
                    }else if(version=="pytorch_112"){
                        cmd="pip install bigdl-nano[pytorch_112,inference]";
                    }else if(version=="pytorch_113"){
                        cmd="pip install bigdl-nano[pytorch,inference]";
                    }
                }else if(inference="inferenceno"){
                    if(version=="pytorch_110"){
                        cmd="pip install bigdl-nano[pytorch_110] -f https://software.intel.com/ipex-whl-stable";
                    }else if(version=="pytorch_111"){
                        cmd="pip install bigdl-nano[pytorch_111]";
                    }else if(version=="pytorch_112"){
                        cmd="pip install bigdl-nano[pytorch_112]";
                    }else if(version=="pytorch_113"){
                        cmd="pip install bigdl-nano[pytorch]";
                    }
                }
            }else if(release=="nightly"){
                if(inference=="inferenceyes"){
                    if(version=="pytorch_110"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch_110,inference] -f https://software.intel.com/ipex-whl-stable";
                    }else if(version=="pytorch_111"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch_111,inference]";
                    }else if(version=="pytorch_112"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch_112,inference]";
                    }else if(version=="pytorch_113"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch,inference]";
                    }
                }else if(inference="inferenceno"){
                    if(version=="pytorch_110"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch_110] -f https://software.intel.com/ipex-whl-stable";
                    }else if(version=="pytorch_111"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch_111]";
                    }else if(version=="pytorch_112"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch_112]";
                    }else if(version=="pytorch_113"){
                        cmd="pip install --pre --upgrade bigdl-nano[pytorch]";
                    }
                }
            }
        }else if(framework="tensorflow"){
            if(inference=="inferenceyes"){
                if (version=="tf2_27"){
                    if (release=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-nano[tensorflow_27,inference]";
                    }else if(release=="stable"){
                        cmd="pip install bigdl-nano[tensorflow_27,inference]";
                    }
                }else if(version=="tf2_28"){
                    if (release=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-nano[tensorflow_28,inference]";
                    }else if(release=="stable"){
                        cmd="pip install bigdl-nano[tensorflow_28,inference]";
                    }
                }else if(version=="tf2_29"){
                    if (release=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-nano[tensorflow,inference]";
                    }else if(release=="stable"){
                        cmd="pip install bigdl-nano[tensorflow,inference]";
                    }
                }else if(version=="tf2_210"){
                    if (release=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-nano[tensorflow_210,inference]";
                    }else if(release=="stable"){
                        cmd="pip install bigdl-nano[tensorflow_210,inference]";
                    }
                }
            }else if(inference=="inferenceno"){
                if(version=="tf2_27"){
                    if(release=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-nano[tensorflow_27]";
                    }else if(release=="stable"){
                        cmd="pip install bigdl-nano[tensorflow_27]";
                    }
                }else if(version=="tf2_28"){
                    if(release=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-nano[tensorflow_28]";
                    }else if(release=="stable"){
                        cmd="pip install bigdl-nano[tensorflow_28]";
                    }
                }else if(version=="tf2_29"){
                    if(release=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-nano[tensorflow]";
                    }else if(release=="stable"){
                        cmd="pip install bigdl-nano[tensorflow]";
                    }
                }else if(version=="tf2_210"){
                    if(release=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-nano[tensorflow_210]";
                    }else if(release=="stable"){
                        cmd="pip install bigdl-nano[tensorflow_210]";
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
        if (framework=="tensorflow"){
            version="tf2_29";
        }else{
            version="pytorch_113";
        }
    }
    else if (releases.indexOf(id)>=0){
        release=id;
    }
    else if (inferences.indexOf(id)>=0){
        inference=id;
    }
    else if (versions.indexOf(id)>=0){
        version=id;
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
