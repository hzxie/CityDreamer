/**
 * @File:   app.js
 * @Author: Haozhe Xie
 * @Date:   2023-06-30 14:08:59
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-07-07 11:19:48
 * @Email:  root@haozhexie.com
 */

String.prototype.format = function() {
    let newStr = this, i = 0
    while (/%s/.test(newStr)) {
        newStr = newStr.replace("%s", arguments[i++])
    }
    return newStr
}

function setUpCanvas(canvas, options) {
    canvas.options = options
    // Set up canvas size
    resetCanvasSize(canvas)
    $(window).on("resize", function() {
        resetCanvasSize(canvas)
    })
    // Set up zoom in and out functions
    if (options["zoomable"]) {
        canvas.hoverCursor = "pointer"
        canvas.on("mouse:wheel", function(opt) {
            let delta = opt.e.deltaY,
                zoom = canvas.getZoom()

            zoom = zoom - delta / 200
            if (zoom > 50) {
                zoom = 50
            }
            if (zoom < 1) {
                zoom = 1
            }
            canvas.zoomToPoint({
                x: opt.e.offsetX,
                y: opt.e.offsetY
            }, zoom)

            if (options["bind:transform"] !== undefined) {
                let scale = options["bind:transform"].width / canvas.width
                options["bind:transform"].zoomToPoint({
                    x: opt.e.offsetX * scale,
                    y: opt.e.offsetY * scale
                }, zoom)
            }
            opt.e.preventDefault()
            opt.e.stopPropagation()
        })
    }
    if (options["bind:transform"]) {
        canvas.on("object:moving", function(e) {
            let currImg = e.target.canvas._objects[0],
                bindImg = options["bind:transform"]._objects[0],
                scale = options["bind:transform"].width / e.target.canvas.width

            if (currImg === undefined || bindImg === undefined ||
                currImg._element.className !== "canvas-img" ||
                bindImg._element.className !== "canvas-img") {
                return
            }
            // Move the image object on the binded canvas
            bindImg.top = currImg.top * scale
            bindImg.left = currImg.left * scale
            bindImg.setCoords()
            options["bind:transform"].renderAll()
        })
    }
    if (options["normalization"]) {
        canvas.on("object:added", function(e) {
            let container = $(e.target.canvas.lowerCanvasEl).parent().parent(),
                imgDrop = $("input[type=file]", container)

            if (e.target._element.className !== "canvas-img") {
                return
            }
            if ($(imgDrop).attr("scale") !== undefined) {
                return
            }
            // Normalize image
            let formData = new FormData()
            formData.append("image", imgDrop[0].files[0])
            $.ajax({
                url: "/img/normalize.action",
                type: "POST",
                data: formData,
                processData: false,
                contentType: false,
            }).done(function(resp) {
                $(imgDrop).attr("scale", resp["scale"])
                canvas.clear()
                fabric.Image.fromURL("/img/get-normalized/" + resp["filename"], function(img) {
                    img.hasControls = img.hasBorders = false
                    img.selectable = false
                    img.scaleToWidth(container.width(), false)
                    canvas.add(img)
                })
            })
        })
    }
}

function resetCanvasSize(canvas) {
    delegateCanvas = canvas.options["delegate"];
    if (delegateCanvas == undefined) {
        delegateCanvas = canvas
    } 
    canvas.options["width"] = delegateCanvas.lowerCanvasEl.parentNode.parentNode.clientWidth - 30
    canvas.options["height"] = canvas.options["width"]
    canvas.setHeight(canvas.options["height"])
    canvas.setWidth(canvas.options["width"])
    canvas.renderAll()
}

$(function() {
    $(".layout.step").addClass("active")
    $(".layout.section").addClass("active")
    // Set up dropdowns
    $(".dropdown").dropdown()
    $("#layout-data-src").on("change", function() {
        $(".five.wide.field", ".layout.section .fields").addClass("hidden")
        $(".two.wide.field", ".layout.section .fields").addClass("hidden")
        $(".imgdrop").addClass("hidden")
        if ($(this).val() == "generator") {
            $(".five.wide.field", ".layout.section .fields").removeClass("hidden")
            $(".two.wide.field", ".layout.section .fields").removeClass("hidden")
        } else if ($(this).val() == "osm") {
            $(".imgdrop").each(function() {
                if (!$(this).hasClass("uploaded")) {
                    $(this).removeClass("hidden")
                }
            })
        }
    })
    // Set up canvas
    segMapCanvas = new fabric.Canvas("seg-map-canvas")
    hfCanvas = new fabric.Canvas("hf-canvas")
    camTrjCanvas = new fabric.Canvas("cam-trj-canvas")
    setUpCanvas(segMapCanvas, {
        "maskable": true,
        "movable": true,
        "zoomable": true,
        "bind:transform": hfCanvas
    })
    setUpCanvas(hfCanvas, {
        "movable": true,
        "zoomable": true,
        "normalization": true,
        "bind:transform": segMapCanvas
    })
    setUpCanvas(camTrjCanvas, {
        "delegate": segMapCanvas,
        "movable": true,
        "zoomable": true,
    })
    // Set up image uploaders
    $("#seg-map-uploader").imgdrop({
        "viewer": segMapCanvas
    })
    $("#hf-uploader").imgdrop({
        "viewer": hfCanvas
    })
})

$(".step").on("click", function() {
    let currentStep = $(this).attr("class").split(" ")[0]
    $(".step").removeClass("active")
    $(".section").removeClass("active")
    $(".%s.step".format(currentStep)).addClass("active")
    $(".%s.section".format(currentStep)).addClass("active")
})
