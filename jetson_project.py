#!/usr/bin/python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import jetson.inference
import jetson.utils

# --model=models/obj_detection2/ssd-mobilenet.onnx, --labels=models/obj_detection2/labels.txt
model = jetson.inference.detectNet(
                                   argv=["--model=obj_detection2/ssd-mobilenet.onnx",
                                         "--labels=obj_detection2/labels.txt",
                                         "--input-blob=input_0",
                                         "--output-cvg=scores",
                                         "--output-bbox=boxes"],
                                         threshold = 0.7
                                  )


camera = jetson.utils.videoSource("/dev/video0") # V4L2 Webcam
display = jetson.utils.videoOutput("display://0") # display in a pop-up window

while display.IsStreaming():
    img = camera.Capture()
    detections = model.Detect(img)
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(model.GetNetworkFPS()))

    for detection in detections:
        item = model.GetClassDesc(detection.ClassID)
        
        if item == "Mask on Face":
            print(fg('green') + item)
        elif item == "Mask":
            print(fg('red') + "Put on your mask!")
            
