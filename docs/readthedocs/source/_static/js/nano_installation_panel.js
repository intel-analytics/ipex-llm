var frameworks = ["pytorch", "tensorflow"];
var versions = ["pytorch_110", "pytorch_111", "pytorch_112", "pytorch_113",
                "tensorflow_27", "tensorflow_28", "tensorflow_29", "tensorflow_210"];
var intel_tensorflow_options = ["intel_tensorflow_yes", "intel_tensorflow_no"]
var inferences = ["inference_yes", "inference_no"];
var releases = ["stable", "nightly"];

var framework = "pytorch";
var version = "pytorch_113";
var intel_tensorflow_option = "intel_tensorflow_yes"
var inference = "inference_yes";
var release = "nightly";

function refresh_cmd(){
    $("#version").empty();
    $("#intel_tensorflow").remove();
    if(framework == "pytorch"){
        $("#version").append("<td colspan='1'>PT Version</td>\
                              <td colspan='1'><button id='pytorch_113' class='install_option_button'>1.13</button></td>\
                              <td colspan='1'><button id='pytorch_112' class='install_option_button'>1.12</button></td>\
                              <td colspan='1'><button id='pytorch_111' class='install_option_button'>1.11</button></td>\
                              <td colspan='1'><button id='pytorch_110' class='install_option_button'>1.10</button></td>");
    }
    else if(framework == "tensorflow"){
        $("#version").append("<td colspan='1'>TF Version</td>\
                              <td colspan='1'><button id='tensorflow_210' class='install_option_button'>2.10</button></td>\
                              <td colspan='1'><button id='tensorflow_29' class='install_option_button'>2.9</button></td>\
                              <td colspan='1'><button id='tensorflow_28' class='install_option_button'>2.8</button></td>\
                              <td colspan='1'><button id='tensorflow_27' class='install_option_button'>2.7</button></td>");

        $("#version").after("<tr id='intel_tensorflow'>\
                                <td colspan='1'>Intel TensorFlow</td>\
                                <td colspan='2'><button id='intel_tensorflow_yes' class='install_option_button'>Yes</button></td>\
                                <td colspan='2'><button id='intel_tensorflow_no' class='install_option_button'>No</button></td>\
                             </tr>");
    }

    reset_color(frameworks);
    reset_color(versions);
    reset_color(intel_tensorflow_options);
    reset_color(inferences);
    reset_color(releases);

    set_color(framework);
    set_color(version);
    set_color(intel_tensorflow_option);
    set_color(inference);
    set_color(release);

    // disable buttons for options that cannot be selected together
    if (release == "stable"){
        disable(["intel_tensorflow_no"])
    } else {
        enable(["intel_tensorflow_no"])
    }

    if (intel_tensorflow_option == "intel_tensorflow_no" &&
        framework == "tensorflow"){
        disable(["stable"])
    } else {
        enable(["stable"])
    }

    // whether to include pre-release version
    var cmd_pre = (release == "nightly") ? " --pre --upgrade" : "";

    // decide on the bigdl-nano install option for framework
    var cmd_options_framework = version;

    if (intel_tensorflow_option == "intel_tensorflow_no"){
        cmd_options_framework = "stock_" + cmd_options_framework;
    }
    // for default version
    if (version == "pytorch_113"){
        cmd_options_framework = "pytorch";
    } else if (version == "tensorflow_29" && intel_tensorflow_option == "intel_tensorflow_yes"){
        cmd_options_framework = "tensorflow";
    }

    // decide on the bigdl-nano install option for including inference or not
    var cmd_options_inference = (inference == "inference_yes") ? ",inference" : "";

    // other options needed at the end of the installation command
    var cmd_other = "";
    if (version == "pytorch_110"){
        cmd_other = " -f https://software.intel.com/ipex-whl-stable";
    }

    var cmd = "pip install" + cmd_pre + " bigdl-nano[" + cmd_options_framework
              + cmd_options_inference + "]" + cmd_other;

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
$(document).on('click',".install_option_button",function(){
    var id = $(this).attr("id");

    if (frameworks.includes(id)){
        framework = id;
        if (framework == "tensorflow"){
            version = "tensorflow_29";
        }else{
            version="pytorch_113";
        }
    }
    else if (versions.includes(id)){
        version = id;
    }
    else if (intel_tensorflow_options.includes(id)){
        intel_tensorflow_option = id;
    }
    else if (inferences.includes(id)){
        inference = id;
    }
    else if (releases.includes(id)){
        release = id;
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
}, "installation-panel-table button");

refresh_cmd();
