//initialize all needed variables(id of buttons)
var functionalities = ["Forecasting","Anomaly","Simulation"];
var models = ["Prophet", "ARIMA","Deep_learning_models","Machine_learning_models"];
var ais = ["pytorch", "tensorflow"];
var oss=["linux", "win"];
var automls=["automlyes", "automlno"];
var hardwares=["singlenode", "cluster"];
var packages=["pypi", "docker"];
var versions=["stable", "nightly"];

//defalut selected burttons' ids
var functionality="Forecasting";
var model="Deep_learning_models";
var ai="pytorch";
var os="linux";
var automl="automlno";
var hardware="singlenode";
var package="pypi";
var version="nightly";

//main func, to decide what is showed in cmd 
function refresh_cmd(){
    //refresh all the buttons
    reset_color(functionalities);
    reset_color(ais);
    reset_color(oss);
    reset_color(automls);
    reset_color(hardwares);
    reset_color(packages);
    reset_color(versions);
    
    set_color(functionality);
    set_color(ai);
    set_color(os);
    set_color(automl);
    set_color(hardware);
    set_color(package);
    set_color(version);
    
    var cmd = "NA";

    //dynamically set 'model' line
    $("#model").empty();
    if(functionality=="Forecasting"){
        $("#model").append("<td colspan='1'>Model</td>\
        <td colspan='1'><button id='Deep_learning_models' style='font-size: 13px;'>Deep learning models</button></td>\
        <td colspan='2'><button id='Prophet'>Prophet</button></td>\
        <td colspan='1'><button id='ARIMA'>ARIMA</button></td>");
    }
    else if(functionality=="Anomaly"){
        $("#model").append("<td colspan='1'>Model</td>\
        <td colspan='2'><button id='Deep_learning_models'>Deep learning models</button>\
        <td colspan='2'><button id='Machine_learning_models'>Machine learning models</button></td>");
    }
    else if(functionality=="Simulation"){
        $("#model").append("<td colspan='1'>Model</td>\
        <td colspan='4'><button id='Deep_learning_models'>Deep learning models</button>");
    }
    reset_color(models);
    set_color(model);

    //enable 'Deep learning framework' when Deep learning models is selected
    if(model!="Deep_learning_models"){
        disable(ais);
    }
    else{
        enable(ais);
    }

    //disable other buttons in cases
    if(package=="docker"){
        disable(functionalities);
        disable(models);
        disable(ais);
        disable(versions);
        disable(oss);
        disable(hardwares);
        disable(automls);
        cmd="Please refer to <a href=' https://github.com/intel-analytics/BigDL/tree/main/docker/chronos-nightly'>docker installation guide.</a>";
    }else if(os=="win"){
        disable(functionalities);
        disable(models);
        disable(ais);
        disable(versions);
        disable(packages);
        disable(hardwares);
        disable(automls);
        cmd="Please refer to <a href='https://bigdl.readthedocs.io/en/latest/doc/Chronos/Howto/windows_guide.html'>windows_guide.</a>";
    }else{
        enable(functionalities);
        enable(models);
        enable(versions);
        enable(oss);
        enable(packages);
        enable(hardwares);
        enable(automls);
        
        //change cmd according to different choices
        if(model=="Deep_learning_models"){
            if(ai=="pytorch"){
                if(automl=="automlno"){
                    if(hardware=="singlenode"){
                        if(version=="nightly"){
                            cmd="pip install --pre --upgrade bigdl-chronos[pytorch]";
                        }else if(version=="stable"){
                            cmd="pip install bigdl-chronos";
                        }
                    }else if(hardware=="cluster"){
                        if(version=="nightly"){
                            cmd="pip install --pre --upgrade bigdl-chronos[pytorch,distributed]";
                        }else if(version=="stable"){
                            cmd="pip install bigdl-chronos[all]";
                        }
                    }
                }else if(automl=="automlyes"){
                    if(hardware=="singlenode"){
                        if(version=="nightly"){
                            cmd="pip install --pre --upgrade bigdl-chronos[pytorch,automl]";
                        }else if(version=="stable"){
                            cmd="pip install bigdl-chronos[all]";
                        }
                    }else if(hardware=="cluster"){
                        if(version=="nightly"){
                            cmd="pip install --pre --upgrade bigdl-chronos[pytorch,distributed,automl]";
                        }else if(version=="stable"){
                            cmd="pip install bigdl-chronos[all]";
                        }
                    }
                }
            }else if(ai=="tensorflow"){
                if(package=="pypi"&&os=="linux"){
                    cmd="Please refer to <a href=' https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html#tensorflow-backend'>tensorflow installation guide.</a>"
                }
            }
        }else if(model=="Prophet"){
            if(automl=="automlno"){
                if(hardware=="singlenode"){
                    if(version=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-chronos; pip install prophet==1.1.0";
                    }else if(version=="stable"){
                        cmd="pip install bigdl-chronos; pip install prophet==1.1.0";
                    }
                }else if(hardware=="cluster"){
                    if(version=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install prophet==1.1.0";
                    }else if(version=="stable"){
                        cmd="pip install bigdl-chronos[all]; pip install prophet==1.1.0";
                    }
                }
            }else if(automl=="automlyes"){
                if(hardware=="singlenode"){
                    if(version=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install prophet==1.1.0";
                    }else if(version=="stable"){
                        cmd="pip install bigdl-chronos[all]; pip install prophet==1.1.0";
                    }
                }else if(hardware=="cluster"){
                    if(version=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install prophet==1.1.0";
                    }else if(version=="stable"){
                        cmd="pip install bigdl-chronos[all]; pip install prophet==1.1.0";
                    }
                }
            }
        }else if(model=="ARIMA"){
            if(automl=="automlno"){
                if(hardware=="singlenode"){
                    if(version=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-chronos; pip install pmdarima==1.8.5";
                    }else if(version=="stable"){
                        cmd="pip install bigdl-chronos; pip install pmdarima==1.8.5";
                    }
                }else if(hardware=="cluster"){
                    if(version=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install pmdarima==1.8.5";
                    }else if(version=="stable"){
                        cmd="pip install bigdl-chronos[all]; pip install pmdarima==1.8.5";
                    }
                }
            }else if(automl=="automlyes"){
                if(hardware=="singlenode"){
                    if(version=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install pmdarima==1.8.5";
                    }else if(version=="stable"){
                        cmd="pip install bigdl-chronos[all]; pip install pmdarima==1.8.5";
                    }
                }else if(hardware=="cluster"){
                    if(version=="nightly"){
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install pmdarima==1.8.5";
                    }else if(version=="stable"){
                        cmd="pip install bigdl-chronos[all]; pip install pmdarima==1.8.5";
                    }
                }
            }
        }else if(model=="Machine_learning_models"){
            if(version=="nightly"){
                cmd="pip install --pre --upgrade bigdl-chronos";
            }else if(version=="stable"){
                cmd="pip install bigdl-chronos";
            }
        }
    }

    $("#cmd").html(cmd);
}

//set the color of selected buttons
function set_color(id){
   $("#"+id).parent().css("background-color","rgb(74, 106, 237)");
   $("#"+id).css("color","white");
   $("#"+id).addClass("isset");
}

//reset the color of unselected buttons
function reset_color(list){
    for (btn in list){
        $("#"+list[btn]).parent().css("background-color","transparent");
        $("#"+list[btn]).css("color","black");
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
        $("#"+list[btn]).parent().css("background-color","rgb(133, 133, 133)");
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

    if (functionalities.indexOf(id)>=0){
        functionality=id;
        model="Deep_learning_models";
    }
    else if (models.indexOf(id)>=0){
        model=id;
    }
    else if (ais.indexOf(id)>=0){
        ai=id;
    }
    else if (oss.indexOf(id)>=0){
        os=id;
    }
    else if (automls.indexOf(id)>=0){
        automl=id;
    }
    else if (hardwares.indexOf(id)>=0){
        hardware=id;
    }
    else if (packages.indexOf(id)>=0){
        package=id;
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
            $(this).parent().css("background-color","rgb(74, 106, 237)");
            $(this).css("color","white");
        }
    },
    mouseleave: function () {
        if(!$(this).hasClass("isset") && $(this).prop("disabled")!=true){
            $(this).parent().css("background-color","transparent");
            $(this).css("color","black");
        }
    }
}, "button");

refresh_cmd();