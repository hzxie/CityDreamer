/**
 * @File:   jquery.image-drop.js
 * @Author: Haozhe Xie
 * @Date:   2023-06-30 15:37:23
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-07-12 14:51:38
 * @Email:  root@haozhexie.com
 */

(function ($) {
    $.fn.imgdrop = function(options) {
        let container = $(this)
        var options = $.extend({
            "width": undefined,
            "height": undefined,
            "viewer": undefined,
            "putUrl": undefined,
            "getUrl": undefined,
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
            let filename = undefined
            if (options["putUrl"] != undefined && options["getUrl"] != undefined) {
                let formData = new FormData()
                formData.append("image", evt.target.files[0])
                $.ajax({
                    "async": false,
                    "url": options["putUrl"],
                    "type": "POST",
                    "data": formData,
                    "processData": false,
                    "contentType": false,
                }).done(function(resp) {
                    filename = resp["filename"]
                })
            }
            if (options["viewer"] !== undefined) {
                options["viewer"].clear()
                if (filename) {
                    options["viewer"].filename = filename
                }

                let imgUrl = filename ?
                             options["getUrl"] + filename :
                             URL.createObjectURL(evt.target.files[0])
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