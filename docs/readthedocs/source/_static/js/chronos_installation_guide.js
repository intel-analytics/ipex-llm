var ais = ["pytorch", "tensorflow", "prophet", "pmdarima"];
var releases=["stable", "nightly"]
var oss=["linux", "win"];
var ways=["pypi", "docker"];
var hardwares=["singlenode", "cluster"];
var automls=["automlyes", "automlno"]
var ai="pytorch";
var os="linux";
var release="nightly";
var way="pypi";
var hardware="singlenode";
var automl="automlno"



function set_color(id){
   $("#"+id).parent().css("background-color","rgb(74, 106, 237)");
   $("#"+id).css("color","white");
}

function reset_color(list){
    for (btn in list){
        $("#"+list[btn]).parent().css("background-color","transparent");
        $("#"+list[btn]).css("color","black");
    }
}

function disable(list){
    for(btn in list){
        $("#"+list[btn]).css("text-decoration","line-through");
        $("#"+list[btn]).attr("disabled","true");
    }
    reset_color(list);
}

function enable(list){
    for(btn in list){
        $("#"+list[btn]).css("text-decoration","none");
        $("#"+list[btn]).attr("disabled",false);
    }
}

function refresh_cmd(){
    reset_color(ais);
    reset_color(oss);
    reset_color(releases);
    reset_color(ways);
    reset_color(hardwares);
    reset_color(automls);
    set_color(ai);
    set_color(os);
    set_color(release);
    set_color(way);
    set_color(hardware);
    set_color(automl);

    var cmd = "NA";

    if(way=="docker"){
        disable(ais);
        disable(releases);
        disable(oss);
        disable(hardwares);
        disable(automls);
        cmd="Please refer to <a href=' https://github.com/intel-analytics/BigDL/tree/main/docker/chronos-nightly'>docker installation guide.</a>";
    }else if(os=="win"){
        disable(ais);
        disable(releases);
        disable(ways);
        disable(hardwares);
        disable(automls);
        cmd="Please refer to <a href='https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/windows_guide.html'>windows_guide.</a>";
    }else{
        enable(ais);
        enable(releases);
        enable(oss);
        enable(ways);
        enable(hardwares);
        enable(automls);
    }


    if(ai=="pytorch"){
        if(way=="pypi"&&os=="linux"){
            if(hardware=="singlenode"){
                if(automl=="automlno"){
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos[pytorch]"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos"
                }else{
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos[pytorch,automl]"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos[all]"
                }
            }
            if(hardware=="cluster"){
                if(automl=="automlno"){
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos[pytorch,distributed]"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos[all]"
                }else{
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos[pytorch,distributed,automl]"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos[all]"
                }
            }
        }
    }
    
    if(ai=="tensorflow"){
        if(way=="pypi"&&os=="linux"){
            cmd="Please refer to <a href=' https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html#tensorflow-backend'>tensorflow installation guide.</a>"
        }
    }

    if(ai=="prophet"){
        if(way=="pypi"&&os=="linux"){
            if(hardware=="singlenode"){
                if(automl=="automlno"){
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos; pip install prophet==1.1.0"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos; pip install prophet==1.1.0"
                }else{
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install prophet==1.1.0"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos[all]; pip install prophet==1.1.0"
                }
            }
            if(hardware=="cluster"){
                if(automl=="automlno"){
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install prophet==1.1.0"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos[all]; pip install prophet==1.1.0"
                }else{
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install prophet==1.1.0"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos[all]; pip install prophet==1.1.0"
                }
            }
        }
    }

    if(ai=="pmdarima"){
        if(way=="pypi"&&os=="linux"){
            if(hardware=="singlenode"){
                if(automl=="automlno"){
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos; pip install pmdarima==1.8.5"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos; pip install pmdarima==1.8.5"
                }else{
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install pmdarima==1.8.5"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos[all]; pip install pmdarima==1.8.5"
                }
            }
            if(hardware=="cluster"){
                if(automl=="automlno"){
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install pmdarima==1.8.5"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos[all]; pip install pmdarima==1.8.5"
                }else{
                    if(release=="nightly")
                        cmd="pip install --pre --upgrade bigdl-chronos[distributed]; pip install pmdarima==1.8.5"
                    if(release=="stable")
                        cmd="pip install bigdl-chronos[all]; pip install pmdarima==1.8.5"
                }
            }
        }
    }
    $("#cmd").html(cmd);
}

function set_ai(id){
    ai=id;
    refresh_cmd();
}

function set_os(id){
    os=id;
    refresh_cmd();
}

function set_rel(id){
    release=id;
    refresh_cmd();
}

function set_way(id){
    way=id;
    refresh_cmd();
}

function set_hardware(id){
    hardware=id;
    refresh_cmd();
}

function set_automl(id){
    automl=id;
    refresh_cmd();
}

$("button").click(function(){
    //alert($(this).attr("id")); 
    var id = $(this).attr("id");
    if (ais.indexOf(id)>=0){
        set_ai(id);
    }

    if (oss.indexOf(id)>=0){
        set_os(id);
    }
    if (releases.indexOf(id)>=0){
        set_rel(id);
    }

    if (ways.indexOf(id)>=0){
        set_way(id);
    }

    if (hardwares.indexOf(id)>=0){
        set_hardware(id);
    }

    if (automls.indexOf(id)>=0){
        set_automl(id);
    }
});

refresh_cmd();
