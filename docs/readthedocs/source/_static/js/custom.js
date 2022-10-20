$(document).ready(function(){
//  $('.btn.dropdown-toggle.nav-item').text('Libraries'); // change text for dropdown menu in header from More to Libraries

    // hide the original left sidebar ads display
    $('#ethical-ad-placement').css({
        "display":"none"
    });

    // manually add the ads to the end of content
    $(".bd-article").append(
        "<br />\
        <div style='display:flex;justify-content:center;'\
        <div\
        id='ethical-ad-placement'\
        class='horizontal'\
        data-ea-publisher='readthedocs'\
        data-ea-type='image'\
      ></div>\
      </div>"
    );

    // make tables responsive
    $("table").wrap(
        "<div style='overflow-x:auto;'></div>"
    );
})