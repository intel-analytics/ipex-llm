$(document).ready(function(){
//    $('.btn.dropdown-toggle.nav-item').text('Libraries'); // change text for dropdown menu in header from More to Libraries
    $('.navbar-end-item.navbar-end__search-button-container').remove(); // remove search button from top bar manually
    $('#ethical-ad-placement').css({
        "transform":"scale(0.3)",
        "position":"absolute",
        "top":"-75px",
        "left":"-110px",
        "opacity":"0.3"
    });
    $('#ethical-ad-placement').parent().css({
        "position":"relative",
        "height":"60px"
    })
})

// #ethical-ad-placement: transform:scale(0.3) position:absolute(前一个设置成relative height 60px) top:-75px left: -15px opacity:0.3