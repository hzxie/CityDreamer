/**
 * @File:   jquery.image-drop.js
 * @Author: Haozhe Xie
 * @Date:   2023-06-30 15:37:23
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-07-07 15:23:49
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

        container.append("<input type='file'>")
        container.on("change", "input[type=file]", function(evt) {
            let imgUrl = URL.createObjectURL(evt.target.files[0])
            if (options["viewer"] !== undefined) {
                options["viewer"].clear()
                fabric.Image.fromURL(imgUrl, function(img) {
                    options["viewer"].setBackgroundImage(
                        img,
                        options["viewer"].renderAll.bind(options["viewer"]), 
                        {
                            originX: 'left',
                            originY: 'top',
                            left: 0,
                            top: 0,
                            scaleX: options["viewer"].width / img.width,
                            scaleY: options["viewer"].height / img.height
                        }
                    )
                    options["viewer"].fire("object:added", {target: img})
                })
            }
            if (options["hideAfterUpload"]) {
                container.addClass("hidden uploaded")
            }
        });
        return this;
    };
}(jQuery));