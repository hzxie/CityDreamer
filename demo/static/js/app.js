/**
 * @File:   app.js
 * @Author: Haozhe Xie
 * @Date:   2023-06-30 14:08:59
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-07-12 10:15:27
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
    canvas.selection = false
    canvas.hoverCursor = "pointer"
    // Set up canvas size
    resetCanvasSize(canvas)
    $(window).on("resize", function() {
        resetCanvasSize(canvas)
    })
    // Detect the mouse up and down events
    canvas.isEditMode = false
    canvas.pointer = {"abs": {"x": 0, "y": 0}, "rel": {"x": 0, "y": 0}}
    canvas.on("mouse:down", function(e) {
        canvas.isMouseDown = true
        // rel is used for dragging backgrounds; abs is used for drawing shapes
        canvas.pointer.rel = e.pointer
        canvas.pointer.abs = canvas.getPointer(e.pointer)
        // Remember the initial positions of objects
        if (canvas.backgroundImage) {
            canvas.backgroundImage.origin = {
                "top": canvas.backgroundImage.top,
                "left": canvas.backgroundImage.left,
            }
        }
        for (let i = 0; i < canvas._objects.length; ++ i) {
            canvas._objects[i].origin = {
                "top": canvas._objects[i].top,
                "left": canvas._objects[i].left
            }
        }
        // Create an initial shape in edit mode
        if (canvas.isEditMode) {
            options = {
                "left": canvas.pointer.abs.x,
                "top": canvas.pointer.abs.y,
                "hasControls": false,
                "hasBorders": false,
                "selectable": false
            }
            if (canvas.shape == "rect") {
                canvas.remove(...canvas.getObjects())
                canvas.add(new fabric.Rect(Object.assign(options, {
                    "width": 0,
                    "height": 0,
                    "fill": "rgba(255, 255, 255, .7)"
                })))
            } else if (canvas.shape == "circle") {
                canvas.remove(...canvas.getObjects())
                canvas.add(new fabric.Circle(Object.assign(options, {
                    "radius": 0,
                    "fill": "rgba(0, 0, 0, 0)",
                    "stroke": "rgba(255, 255, 255, .5)",
                    "strokeWidth": 4
                })))
            } else if (canvas.shape == "line") {
                canvas.remove(...canvas.getObjects())
                canvas.add(new fabric.Line([
                    options["left"], options["top"], options["left"], options["top"]
                ], Object.assign(options, {
                    "stroke": "rgba(255, 255, 255, .5)",
                    "strokeWidth": 4
                })))
            } else if (canvas.shape == "polyline") {
                if (canvas._objects[0].points === undefined) {
                    canvas.remove(...canvas.getObjects())
                }
                canvas.add(new fabric.Polyline([
                    {"x": options["left"], "y": options["top"]},
                    {"x": options["left"], "y": options["top"]}
                ], Object.assign(options, {
                    "fill": "rgba(0, 0, 0, 0)",
                    "stroke": "rgba(255, 255, 255, .5)",
                    "strokeWidth": 4,
                    "objectCaching": false
                })))
            }
        }
    })
    canvas.on("mouse:up", function(e) {
        canvas.isMouseDown = false
        // abs is used for drawing shapes
        canvas.pointer.abs = canvas.getPointer(e.pointer)
        if (canvas.isEditMode) {
            if (canvas.shape == "polyline") {
                let uniqueObject = canvas._objects[0],
                    nPoints = uniqueObject.points.length

                uniqueObject.points[nPoints - 1] = canvas.pointer.abs
                // Avoid appending duplicate points
                if (nPoints >= 2) {
                    let lastPoint = uniqueObject.points[nPoints - 1],
                        last2Point = uniqueObject.points[nPoints - 2]

                    if (lastPoint.x != last2Point.x || lastPoint.y != last2Point.y) {
                        uniqueObject.points.push(canvas.pointer.abs)
                    }
                }
            }
        }
        canvas.renderAll()
    })
    // Sync the background image to another canvas
    if (options["bind:destination"]) {
        canvas.on("object:added", function(e) {
            let element = e.target._element ? e.target._element.className : ""
            if (element === "canvas-img") {
                fabric.Image.fromURL(e.target.getSrc(), function(img) {
                    options["bind:destination"].setBackgroundImage(
                        img,
                        options["bind:destination"].renderAll.bind(options["bind:destination"]),
                        {
                            originX: 'left',
                            originY: 'top',
                            left: 0,
                            top: 0,
                            scaleX: options["bind:destination"].width / img.width,
                            scaleY: options["bind:destination"].height / img.height
                        }
                    )
                })
            }
        })
    }
    // Set up zoom in and out functions
    if (options["zoomable"]) {
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
        // Set up handlers for dragging in the background
        canvas.on("mouse:move", function(e) {
            if (canvas.isEditMode || !canvas.isMouseDown || !canvas.backgroundImage) {
                return
            }
            let zoom = canvas.getZoom(),
                // DO NOT USE canvas.getPointer() here
                pointer = e.pointer,
                deltaX = (pointer.x - canvas.pointer.rel.x) / zoom,
                deltaY = (pointer.y - canvas.pointer.rel.y) / zoom

            canvas.backgroundImage.top = canvas.backgroundImage.origin.top + deltaY
            canvas.backgroundImage.left = canvas.backgroundImage.origin.left + deltaX
            for (let i = 0; i < canvas._objects.length; ++ i) {
                canvas._objects[i].top = canvas._objects[i].origin.top + deltaY
                canvas._objects[i].left = canvas._objects[i].origin.left + deltaX
            }
            canvas.renderAll()

            if (options["bind:transform"]) {
                let currImg = canvas.backgroundImage,
                    bindImg = options["bind:transform"].backgroundImage,
                    scale = options["bind:transform"].width / canvas.width
                
                if (currImg === null || bindImg === null) {
                    return
                }
                bindImg.top = currImg.top * scale
                bindImg.left = currImg.left * scale
                options["bind:transform"].renderAll()
            }
        })
    }
    if (options["editable"]) {
        let container = $(canvas.lowerCanvasEl).parent()
        $(container).append("<span class='mode hidden'>Edit Mode</span>")

        $(window).on("keydown", function(e) {
            let key = e.keyCode || e.which,
                expectedKeys = [17, 91, 93, 224]
            if (expectedKeys.includes(key)) { // CTRL / Command is pressed
                $(".mode", container).removeClass("hidden")
                canvas.isEditMode = true
                canvas.forEachObject(function(o) {o.evented = false; o.selectable = false})
            }
        })
        $(window).on("keyup", function(e) {
            let key = e.keyCode || e.which,
                expectedKeys = [17, 91, 93, 224]
            if (expectedKeys.includes(key)) { // CTRL / Command is pressed
                $(".mode", container).addClass("hidden")
                canvas.isEditMode = false
                canvas.forEachObject(function(o) {o.evented = true; o.selectable = true})
            }
        })
    }
    if (options["drawable"]) {
        canvas.on("mouse:move", function(e) {
            if (!canvas.isEditMode || !canvas.isMouseDown || !canvas.backgroundImage) {
                return
            }
            let uniqueObject = canvas._objects[0],
                currentPointer = canvas.getPointer(e.pointer)

            // Update canvas shape on mouse move
            if (canvas.shape === "polyline") {
                uniqueObject.points[uniqueObject.points.length - 1] = currentPointer
            } else if (canvas.shape) {
                if (canvas.pointer.abs.x > currentPointer.x) {
                    uniqueObject.set({left: Math.abs(currentPointer.x)})
                }
                if (canvas.pointer.abs.y > currentPointer.y){
                    uniqueObject.set({top: Math.abs(currentPointer.y)})
                }
                if (canvas.shape === "rect") {
                    uniqueObject.set({
                        "width": Math.abs(canvas.pointer.abs.x - currentPointer.x),
                        "height": Math.abs(canvas.pointer.abs.y - currentPointer.y)
                    })
                } else if (canvas.shape === "circle") {
                    let deltaX = canvas.pointer.abs.x - currentPointer.x,
                        deltaY = canvas.pointer.abs.y - currentPointer.y
    
                    uniqueObject.set({
                        "radius": Math.sqrt(deltaX * deltaX + deltaY * deltaY) / Math.sqrt(2) / 2
                    })
                } else if (canvas.shape === "line") {
                    uniqueObject.set({
                        "x2": currentPointer.x,
                        "y2": currentPointer.y
                    })
                }
            }
            canvas.renderAll()
        })
    }
    if (options["normalization"]) {
        canvas.on("object:added", function(e) {
            let container = $(e.target.canvas.lowerCanvasEl).parent().parent(),
                imgDrop = $("input[type=file]", container),
                element = e.target._element ? e.target._element.className : ""

            if (element !== "canvas-img") {
                return
            }
            if ($(imgDrop).attr("scale") !== undefined) {
                return
            }
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
                fabric.Image.fromURL("/img/get-normalized/" + resp["filename"], function(img) {
                    canvas.setBackgroundImage(
                        img,
                        canvas.renderAll.bind(canvas),
                        {
                            originX: 'left',
                            originY: 'top',
                            left: 0,
                            top: 0,
                            scaleX: canvas.width / img.width,
                            scaleY: canvas.height / img.height
                        }
                    )
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

// Set up active step on the left
$(".layout.step").addClass("active")
$(".layout.section").addClass("active")
$(".step").on("click", function() {
    let currentStep = $(this).attr("class").split(" ")[0]
    $(".step").removeClass("active")
    $(".section").removeClass("active")
    $(".%s.step".format(currentStep)).addClass("active")
    $(".%s.section".format(currentStep)).addClass("active")
})

// Set up sliders
$("#camera-altitude").slider({
    min: 128,
    max: 778,
    smooth: true,
    onChange: function() {
        $(".value", this).html($(this).slider("get value"))
    }
})
$("#camera-altitude").slider("set value", 353)
$("#elevation-altitude").slider({
    min: 30,
    max: 60,
    smooth: true,
    onChange: function() {
        $(".value", this).html($(this).slider("get value"))
    }
})
$("#elevation-altitude").slider("set value", 45)

// Set up canvas
segMapCanvas = new fabric.Canvas("seg-map-canvas")
hfCanvas = new fabric.Canvas("hf-canvas")
camTrjCanvas = new fabric.Canvas("cam-trj-canvas")
setUpCanvas(segMapCanvas, {
    "drawable": true,
    "editable": true,
    "zoomable": true,
    "bind:transform": hfCanvas,
    "bind:destination": camTrjCanvas
})
setUpCanvas(hfCanvas, {
    "zoomable": true,
    "normalization": true,
    "bind:transform": segMapCanvas
})
setUpCanvas(camTrjCanvas, {
    "delegate": segMapCanvas,
    "drawable": true,
    "editable": true,
    "zoomable": false,
})
    
// Set up image uploaders
$("#seg-map-uploader").imgdrop({
    "viewer": segMapCanvas
})
$("#hf-uploader").imgdrop({
    "viewer": hfCanvas
})

// Set up initial shape in canvas
segMapCanvas.shape = "rect"
$("#trajectory-mode").on("change", function() {
    $(".red.button", ".trajectory.section").addClass("hidden")
    if ($(this).val() == "orbit") {
        camTrjCanvas.shape = "circle"
    } else if ($(this).val() == "p2p") {
        camTrjCanvas.shape = "line"
    } else if ($(this).val() == "keypoints") {
        camTrjCanvas.shape = "polyline"
        $(".red.button", ".trajectory.section").removeClass("hidden")
    }
})

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

// Set up events on button clicks
$(".red.button", ".trajectory.section").on("click", function(e) {
    e.preventDefault()
    if (camTrjCanvas.shape !== "polyline")  {
        return
    }

    let uniqueObject = camTrjCanvas._objects[0]
    if (uniqueObject) {
        if (uniqueObject.points.length <= 2) {
            camTrjCanvas.remove(...camTrjCanvas.getObjects())
        } else {
            let nPoints = uniqueObject.points.length,
                lastPoint = uniqueObject.points[nPoints - 1],
                last2Point = uniqueObject.points[nPoints - 2]

            if (lastPoint.x == last2Point.x && lastPoint.y == last2Point.y) {
                // Remove duplicated points
                uniqueObject.points.pop()
            }
            uniqueObject.points.pop()
        }
        camTrjCanvas.renderAll()
    }
})
