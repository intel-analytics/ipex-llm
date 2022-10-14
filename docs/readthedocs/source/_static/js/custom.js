$(document).ready(function(){
    $('.btn.dropdown-toggle.nav-item').text('Libraries'); // change text for dropdown menu in header from More to Libraries
    $('.navbar-end-item.navbar-end__search-button-container').remove(); // remove search button from top bar manually
    $('#ethical-ad-placement').css({  //weanken the display effect of ads div
        // "transform":"scale(0.3)",   // make it smaller
        // "position":"absolute",      // modify its position
        // "top":"-75px",
        // "left":"-110px",
        // "opacity":"0.7"             // modify its opacity
        "display":"none"
    });
    $('#ethical-ad-placement').parent().css({
        "position":"relative",      // modify ads position
        "height":"60px"             // give it a default height to prevent wrong display
    })
})
