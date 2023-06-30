/**
 * @File:   jquery.image-drop.js
 * @Author: Haozhe Xie
 * @Date:   2023-06-30 15:37:23
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-06-30 16:49:08
 * @Email:  root@haozhexie.com
 */

(function ($) {
    $.fn.imgdrop = function(options) {
        let container = $(this)
        var options = $.extend({
            "width": undefined,
            "height": undefined,
            "viewer": undefined,
            "url": undefined,
            "hideAfterUpload": true
        }, options)

        if (options["width"] != undefined) {
            container.css("width", options["width"])
        }
        if (options["height"] != undefined) {
            container.css("height", options["height"])
        } else {
            container.css("height", container.width())
        }

        container.append("<input type='file'>");
        container.on("change", "input[type=file]", function(evt) {
            let imgUrl = URL.createObjectURL(evt.target.files[0])
            if (options["viewer"] !== undefined) {
                options["viewer"].clear()
                fabric.Image.fromURL(imgUrl, function(img) {
                    img.hasControls = img.hasBorders = false
                    img.selectable = false
                    img.scaleToWidth(container.width(), false)
                    options["viewer"].add(img)
                })
            }
            if (options["hideAfterUpload"]) {
                container.addClass("hidden")
            }
        });
        return this;
    };
}(jQuery));