/**
 * @File:   app.js
 * @Author: Haozhe Xie
 * @Date:   2023-06-30 14:08:59
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-07-15 08:10:13
 * @Email:  root@haozhexie.com
 */

const CONSTANTS = {
    "MAX_PATCH_SIZE": 768
}

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
                if (canvas._objects[0] && canvas._objects[0].points === undefined) {
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
                    let targetCanvas = options["bind:destination"],
                        scaleX = (img.width - CONSTANTS["MAX_PATCH_SIZE"] * 4) / img.width,
                        scaleY = (img.height - CONSTANTS["MAX_PATCH_SIZE"] * 4) / img.height

                    if (scaleX <= 0 || scaleY <= 0) {
                        console.log("[ERROR] The image is smaller than 3072x3072.")
                        return
                    }
                    targetCanvas.setBackgroundImage(
                        img,
                        targetCanvas.renderAll.bind(targetCanvas),
                        {
                            originX: 'left',
                            originY: 'top',
                            left: 0,
                            top: 0,
                            scaleX: targetCanvas.width / img.width,
                            scaleY: targetCanvas.height / img.height
                        }
                    )
                    targetCanvas.zoomToPoint(
                        new fabric.Point(targetCanvas.width / 2, targetCanvas.height / 2),
                        targetCanvas.getZoom() / Math.max(scaleX, scaleY)
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
            let element = e.target._element ? e.target._element.className : ""

            if (element !== "canvas-img" || canvas.filename === undefined || canvas.normalized) {
                return
            }
            $.get({
                url: "/image/%s/normalize.action".format(canvas.filename),
            }).done(function(resp) {
                fabric.Image.fromURL("/image/%s".format(resp["filename"]), function(img) {
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
function updateMinElevation() {
    let altitude = $("#cam-altitude").slider("get value"),
        elevation = $("#cam-elevation").slider("get value")
        minElevation = Math.ceil(Math.atan(altitude / CONSTANTS["MAX_PATCH_SIZE"]) * 180 / Math.PI)

    if (elevation < minElevation) {
        $("#cam-elevation").slider("set value", minElevation)
    }
}
$("#cam-step-size").slider({
    min: 1,
    max: 20,
    smooth: true,
    onChange: function() {
        $(".value", this).html($(this).slider("get value"))
    }
})
$("#cam-elevation").slider({
    min: 30,
    max: 60,
    smooth: true,
    onChange: function() {
        updateMinElevation()
        $(".value", this).html($(this).slider("get value"))
    }
})
$("#cam-altitude").slider({
    min: 128,
    max: 778,
    smooth: true,
    onChange: function() {
        updateMinElevation()
        $(".value", this).html($(this).slider("get value"))
    }
})
// Set default values for sliders
$("#cam-step-size").slider("set value", 10)
$("#cam-elevation").slider("set value", 45)
$("#cam-altitude").slider("set value", 353)

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
    "viewer": segMapCanvas,
    "putUrl": "/image/upload.action",
    "getUrl": "/image/"
})
$("#hf-uploader").imgdrop({
    "viewer": hfCanvas,
    "putUrl": "/image/upload.action",
    "getUrl": "/image/"
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

// Set up events on button clicks
function getTrajectory() {
    let trajectory = [],
        altitude = $("#cam-altitude").slider("get value"),
        elevation = $("#cam-elevation").slider("get value"),
        camStepSize = $("#cam-step-size").slider("get value"),
        uniqueObject = camTrjCanvas._objects[0],
        scale = camTrjCanvas.backgroundImage ? camTrjCanvas.backgroundImage.scaleX : 1
    
    if (uniqueObject && uniqueObject.type === "circle") {
        trajectory = getOrbitTrajectory(uniqueObject, altitude, elevation, camStepSize, scale)
    } else if (uniqueObject && uniqueObject.type === "line") {
        trajectory = getP2PTrajectory(uniqueObject, altitude, elevation, camStepSize, scale)
    } else if (uniqueObject && uniqueObject.type === "polyline") {
        trajectory = getKeypointsTrajectory(uniqueObject, altitude, elevation, camStepSize, scale)
    }
    for (let i = 0; i < trajectory.length; ++ i) {
        trajectory[i]["camera"]["x"] /= scale
        trajectory[i]["camera"]["y"] /= scale
        trajectory[i]["target"]["x"] /= scale
        trajectory[i]["target"]["y"] /= scale
    }
    console.log(camTrjCanvas.backgroundImage, trajectory)
    return trajectory
}

function getOrbitTrajectory(object, altitude, elevation, stepSize, scale) {
    let cx = object.left + object.radius,
        cy = object.top + object.radius,
        perimeter = 2 * Math.PI * object.radius,
        targetDist = altitude / Math.tan(elevation / 180 * Math.PI) * scale,
        nPoints = Math.round(perimeter / stepSize / 4) * 4,
        trajectory = []

    for (let i = 0; i < nPoints; ++ i) {
        let theta = 2 * Math.PI / nPoints * i,
            camX = cx + object.radius * Math.cos(theta),
            camY = cy + object.radius * Math.sin(theta),
            targetX = targetDist > object.radius ?
                      cx :
                      cx + (object.radius - targetDist) * Math.cos(theta),
            targetY = targetDist > object.radius ?
                      cy :
                      cy + (object.radius - targetDist) * Math.sin(theta)

        trajectory.push({
            "camera": {"x": camX, "y": camY, "z": altitude},
            "target": {"x": targetX, "y": targetY, "z": 0}
        })
    }
    return trajectory
}

function getP2PTrajectory(object, altitude, elevation, stepSize, scale) {
    let deltaX = object.x1 - object.x2,
        deltaY = object.y1 - object.y2,
        dist = Math.sqrt(deltaX * deltaX + deltaY * deltaY),
        targetDist = altitude / Math.tan(elevation / 180 * Math.PI) * scale,
        nPoints = Math.round(dist / stepSize),
        theta = deltaX == 0 ? Math.PI / 2 : Math.atan(deltaY / deltaX),
        signX = object.x1 < object.x2 ? 1 : -1,
        signY = object.y1 < object.y2 ? -1 : 1,
        trajectory = []

    theta = theta < Math.PI / 2 ? theta : Math.PI - theta
    for (let i = 0; i <= nPoints; ++ i) {
        let camX = object.x1 + signX * dist / nPoints * i * Math.cos(theta),
            camY = object.y1 + signY * dist / nPoints * i * Math.sin(theta),
            targetX = camX + signX * targetDist * Math.cos(theta),
            targetY = camY + signY * targetDist * Math.sin(theta)

        trajectory.push({
            "camera": {"x": camX, "y": camY, "z": altitude},
            "target": {"x": targetX, "y": targetY, "z": 0}
        })
    }
    return trajectory
}

function getKeypointsTrajectory(object, altitude, elevation, stepSize, scale) {
    console.log(object, stepSize)
    return []
}

$(".primary.button", ".trajectory.section").on("click", function(e) {
    $(".primary.button", ".trajectory.section").html("Please wait ...")
    $(".primary.button", ".trajectory.section").attr("disabled", "disabled")
    $(".message", ".trajectory.section").addClass("hidden")
    e.preventDefault()

    let errorMessage = "",
        trajectory = getTrajectory(),
        segFileName = segMapCanvas.filename,
        hfFileName = hfCanvas.filename
    if (segFileName === undefined || hfFileName === undefined) {
        errorMessage = "Please generate Segmentation Map and Height Field first."
    } else if (trajectory.length == 0) {
        errorMessage = "Please draw the camera trajectory on Camera Trajectory Configurator."
    }
    if (errorMessage) {
        $(".message", ".trajectory.section").html(errorMessage)
        $(".message", ".trajectory.section").removeClass("hidden")
        $(".primary.button", ".trajectory.section").html("Preview Trajectory")
        $(".primary.button", ".trajectory.section").removeAttr("disabled")
        return
    }
    $.post({
        "url": "/trajectory/preview.action",
        "data": {
            "hf": hfFileName,
            "seg": segFileName,
            "trajectory": JSON.stringify(trajectory)
        },
        "dataType": "json"
    }).done(function(resp) {
        if (resp["filename"]) {
            $("video", ".modal").attr("src", "/video/%s".format(resp["filename"]))
            $(".ui.basic.modal").modal("show")
        } else {
            errorMessage = "Error occurred while rendering the video."
            $(".message", ".trajectory.section").html(errorMessage)
            $(".message", ".trajectory.section").removeClass("hidden")
        }
        $(".primary.button", ".trajectory.section").html("Preview Trajectory")
        $(".primary.button", ".trajectory.section").removeAttr("disabled")
    })
})

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

function waitForVideoRendering(videoName, nFrames) {
    let currentFrame = 0,
        imgRefreshInterval = setInterval(function() {
            $.get({
                url: "/image/%s/%s".format(videoName, currentFrame),
                xhrFields: {responseType: "blob"},
                success: function (imageBlob) {
                    let imgUrl = URL.createObjectURL(imageBlob)
                    $("img", ".image-viewer").addClass("hidden")
                    $(".image-viewer").append("<img src='%s'>".format(imgUrl))
                    if (++ currentFrame >= nFrames) {
                        clearInterval(imgRefreshInterval)
                        $(".primary.button", ".render.section").html("Render Video")
                        $(".primary.button", ".render.section").removeAttr("disabled")
                        $(".green.button", ".render.section").removeAttr("disabled")
                    }
                    $(".progress", ".render.section").progress({percent: currentFrame / nFrames * 100})
                }
            })
        }, 5000)
}

$(".primary.button", ".render.section").on("click", function(e) {
    $(".primary.button", ".render.section").html("Please wait ...")
    $(".primary.button", ".render.section").attr("disabled", "disabled")
    $(".green.button", ".render.section").attr("disabled", "disabled")
    $(".message", ".render.section").addClass("hidden")
    e.preventDefault()

    let errorMessage = "",
        trajectory = getTrajectory(),
        segFileName = segMapCanvas.filename,
        hfFileName = hfCanvas.filename
    if (segFileName === undefined || hfFileName === undefined) {
        errorMessage = "Please generate Segmentation Map and Height Field first."
    } else if (trajectory.length == 0) {
        errorMessage = "Please draw the camera trajectory on Camera Trajectory Configurator."
    }
    if (errorMessage) {
        $(".message", ".render.section").html(errorMessage)
        $(".message", ".render.section").removeClass("hidden")
        $(".primary.button", ".render.section").html("Render")
        $(".primary.button", ".render.section").removeAttr("disabled")
        return
    }
    $.post({
        "url": "/city/render.action",
        "data": {
            "hf": hfFileName,
            "seg": segFileName,
            "trajectory": JSON.stringify(trajectory)
        },
        "dataType": "json"
    }).done(function(resp) {
        waitForVideoRendering(resp["video"], resp["frames"])
    })
})

// Global variables for play/pause videos
playFrameIdx = 0
$(".green.button", ".render.section").on("click", function(e) {
    let images = $("img", ".image-viewer"),
        currentText = $(".green.button", ".render.section").html()
    if (currentText == "Play Video") {
        $(".green.button", ".render.section").html("Pause")
        playInterval = setInterval(function() {
            if (playFrameIdx >= images.length) {
                playFrameIdx = 0
            }
            $("img", ".image-viewer").addClass("hidden")
            $(images[playFrameIdx ++]).removeClass("hidden")
            $(".progress", ".render.section").progress({percent: playFrameIdx / images.length * 100})
        }, 1000)
    } else if (currentText == "Pause") {
        $(".green.button", ".render.section").html("Play Video")
        clearInterval(playInterval)
    }
})
