/**
 * @File:   app.js
 * @Author: Haozhe Xie
 * @Date:   2023-06-30 14:08:59
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-06-30 18:23:49
 * @Email:  root@haozhexie.com
 */

String.prototype.format = function() {
    let newStr = this, i = 0
    while (/%s/.test(newStr)) {
        newStr = newStr.replace("%s", arguments[i++])
    }
    return newStr
}

function resetCanvasSize(canvas, width, height) {
    if (width == undefined) {
        width = canvas.lowerCanvasEl.parentNode.parentNode.clientWidth - 30
    }
    if (height == undefined) {
        height = width
    }
    canvas.setHeight(height)
    canvas.setWidth(width)
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
            $(".imgdrop").removeClass("hidden")
        }
    })
    // Set up canvas
    segMapCanvas = new fabric.Canvas("seg-map-canvas")
    hfCanvas = new fabric.Canvas("hf-canvas")
    resetCanvasSize(segMapCanvas)
    resetCanvasSize(hfCanvas)
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
