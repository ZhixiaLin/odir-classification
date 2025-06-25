
// common scripts

(function () {
   "use strict";

   // Sidebar toggle

   jQuery('.menu-list > a').click(function () {

      var parent = jQuery(this).parent();
      var sub = parent.find('> ul');

      if (!jQuery('body').hasClass('sidebar-collapsed')) {
         if (sub.is(':visible')) {
            sub.slideUp(300, function () {
               parent.removeClass('nav-active');
               jQuery('.body-content').css({ height: '' });
               adjustMainContentHeight();
            });
         } else {
            visibleSubMenuClose();
            parent.addClass('nav-active');
            sub.slideDown(300, function () {
               adjustMainContentHeight();
            });
         }
      }
      return false;
   });

   function visibleSubMenuClose() {

      jQuery('.menu-list').each(function () {
         var t = jQuery(this);
         if (t.hasClass('nav-active')) {
            t.find('> ul').slideUp(300, function () {
               t.removeClass('nav-active');
            });
         }
      });
   }

   function adjustMainContentHeight() {

      // Adjust main content height
      var docHeight = jQuery(document).height();
      if (docHeight > jQuery('.body-content').height())
         jQuery('.body-content').height(docHeight);
   }

   // add class mouse hover

   jQuery('.side-navigation > li').hover(function () {
      jQuery(this).addClass('nav-hover');
   }, function () {
      jQuery(this).removeClass('nav-hover');
   });






})(jQuery);
function loadFile(event, imgId) {
   var reader = new FileReader();
   reader.onload = function () {
      var output = document.getElementById(imgId);
      output.src = reader.result;
   };
   reader.readAsDataURL(event.target.files[0]);
}

document.getElementById('left-eye-image').addEventListener('click', function () {
   document.getElementById('left-eye-upload').click();
});

document.getElementById('right-eye-image').addEventListener('click', function () {
   document.getElementById('right-eye-upload').click();
});
function previewImage(input, previewId) {
   const preview = document.getElementById(previewId);
   if (input.files && input.files[0]) {
      const reader = new FileReader();
      reader.onload = function (e) {
         preview.src = e.target.result;
      };
      reader.readAsDataURL(input.files[0]);
   }
}