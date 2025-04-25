#!/home/<user>/CAMERA/PYTHON/VirEnv/bin/python

# 0) Configure lines containing <user> with settings for your environment.
#
# 1) sudo apt update && sudo apt full-upgrade
#    sudo apt install imx500-all
#    sudo apt install python3-opencv
#    sudo reboot
#
# 2) Build a venv:
#    python3 -m venv /home/<user>/CAMERA/PYTHON/VirEnv --system-site-packages
#
# 3) Operate from the venv:
#    source ~/CAMERA/PYTHON/VirEnv/bin/activate
#
# 4) Set your rpk 'model' of choice below.
#
# 5) Ensure piedgecctv.py is executable:
#    chmod 755 piedgecctv.py
#
# 6) Run:
#    ./piedgecctv.py


# Version 0.3.
# 
# PiEdgeCCTV is an edge AI CCTV system for Raspberry Pi using the IMX500 AI Camera.
# 
# It records video when a matched inference occurs providing combined pre-roll, event and post-roll capture.
# 
# Heavily based on https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py
# 
# See also the incredibly helpful Picamera2 Library manual https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
# 
# Plenty of room for improvement including post-processing framework, am not proud of the number of global variables.
# 
# Prototyped using a Raspberry Pi 5 16GB. IMX500 AI Camera info https://www.raspberrypi.com/documentation/accessories/ai-camera.html
# 
# Info on models https://github.com/raspberrypi/imx500-models/blob/main/README.md
# 
# Forum thread https://forums.raspberrypi.com/viewtopic.php?t=383752
#
# You might wish to increase the default Contiguous Memory Allocator (CMA) value for your Pi, see Picamera2 Library manual.
# 
# It is known 'Network Firmware Upload' sometimes fails, just try again.


import datetime
import time
import os
import subprocess
import cv2
import numpy as np

from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput2
from picamera2.outputs import PyavOutput

from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection

from libcamera import Transform

from functools import lru_cache

# Memory location
mem_folder = "/dev/shm"

# Final destination for video files
output_folder = "/home/<user>/VIDCAPTURE"

# ffmpeg concat text file
text_file_shm = '/dev/shm/input_list.txt'

# Start in non-recording mode
recording = False

# Time difference set to zero
ltime = 0

# AI
model = '/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk'
threshold = 0.65
iou = 0.65
max_detections = 5

# What we want to record
interested = ('person', 'cat', 'dog')

# Frames per second
fps = 5

# Output format
opformat = 'mp4'

# Main stream, assigned to recording, width & height (from align_configuration)
mainw = 2016
mainh = 1520

# Lores stream, assigned to preview, width & height (from align_configuration)
loresw = 640
loresh = 480

# Timestamp
tscolour = (23, 170, 252) # BGR
tsfont = cv2.FONT_HERSHEY_SIMPLEX
# Lores (preview)
tslscale = 0.8
tslorigin = (10, 30)
tslthickness = 2
# Main (recording)
tsmscale = 1.9
tsmorigin = (30, 70)
tsmthickness = 3

# CircularOutput2 buffer duration in ms (pre-roll)
preroll = 7000

# Post roll duration in s (post-roll)
postroll = 8.0

last_detections = []


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    
    # Could be tidied up for this requirement to remove unused code
    
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)


    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
        
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def apply_timestamp(request):

    ts = time.strftime("%Y-%m-%d %X")
    with MappedArray(request, "main") as m:
    
        cv2.putText(m.array, ts, tsmorigin, tsfont, tsmscale, tscolour, tsmthickness)
        
    with MappedArray(request, "lores") as n:
    
        cv2.putText(n.array, ts, tslorigin, tsfont, tslscale, tscolour, tslthickness)
        

def vidrecord():

    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    
    hit = False
           
    for detection in detections:
        label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
    
        lbl = labels[int(detection.category)]
        
        for i in interested:
            if i == lbl:
                hit = True
                break
    
    global recording
    global ltime
    global circular
    global ip, op, op_pre, op_pre_start_fix
    
    if not recording:
        now = datetime.datetime.now()
        nowform = now.strftime('%Y-%m-%d_%Hh%Mm%Ss')
        op_pre = os.path.join(mem_folder, nowform) + "_pre." + opformat
        op_pre_start_fix = os.path.join(mem_folder, nowform) + "_pre_start_fix." + opformat
        op_post = os.path.join(mem_folder, nowform) + "_post." + opformat
        ip = (op_pre_start_fix, op_post)
        os.makedirs(output_folder, exist_ok = True)
        op = os.path.join(output_folder, nowform) + "." + opformat
                   
    if hit and not recording:    # hit = True and recording = False, we need to start recording 
        print("New recording started at:", nowform)
        circular.open_output(PyavOutput(f"{op_pre}", format=opformat))
        picam2.stop_encoder()
        output = PyavOutput(f"{op_post}", format=opformat)
        picam2.start_recording(encoder, output)
        recording = True
        ltime = time.time()
        
    elif hit and recording:    # hit = True and recording = True, make sure ltime is the last frame
        ltime = time.time()
        
    elif recording:  # hit = false and recording = True, we need to stop recording
        if time.time() - ltime > postroll:
            picam2.stop_encoder()
            recording = False
            circular = CircularOutput2(buffer_duration_ms=preroll)
            picam2.start_recording(encoder, circular)
            vid_start_fix(op_pre, op_pre_start_fix)
            concatenate_vids(ip, op)
       

def vid_start_fix(faulty, fixed):
    
    # The flushed CircularOutput2 video has a start value which causes problems
    command = [
        "ffmpeg",
        "-i", faulty,
        "-metadata", "start=0.000000",
        "-c", "copy",
        fixed,
    ]
    
    subprocess.run(command, check=True, capture_output=True, text=True)
    

def concatenate_vids(input_files, output_file):
    """Concatenates two or more video files using ffmpeg."""

    # Create a temporary file list for ffmpeg
    with open(text_file_shm, "w") as f:
        for file in input_files:
            f.write(f"file '{file}'\n")
        f.close()

    # Construct the ffmpeg command
    command = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",  # Allow unsafe file paths (if needed)
        "-i", text_file_shm,
        "-c", "copy",  # Copy streams without re-encoding
        output_file,
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)

    # Clean up temporary files
    os.remove(text_file_shm)
    os.remove(op_pre)
    for z in input_files:
        os.remove(z)



if __name__ == "__main__":

    imx500 = IMX500(model)
    intrinsics = imx500.network_intrinsics
    
    picam2 = Picamera2(imx500.camera_num)
    
    main  = {'size': (mainw, mainh), 'format': 'XRGB8888'}
    lores = {'size': (loresw, loresh), 'format': 'XRGB8888'}  # RPi5

    camcontrols = {'FrameRate': fps}
    
    # Might turn the camera upside down later so leaving in transform
    config = picam2.create_video_configuration(main, lores=lores, controls=camcontrols, buffer_count=12, encode="main", display="lores", transform=Transform(hflip=False, vflip=False))

    encoder = H264Encoder(bitrate=10000000)

    imx500.show_network_fw_progress_bar()
    
    picam2.pre_callback = apply_timestamp

    #picam2.start_preview(Preview.QT, x=2, y=64)
    picam2.start_preview(Preview.QTGL, x=2, y=64)

    picam2.start(config, show_preview=True)
    
    circular = CircularOutput2(buffer_duration_ms=preroll)

    picam2.start_recording(encoder, circular)
    
    
    last_results = None

    while True:
     
        last_results = parse_detections(picam2.capture_metadata())
        
        vidrecord()
 
