#!/home/<user>/CAMERA/PYTHON/VirEnv/bin/python


# 1) sudo apt update && sudo apt full-upgrade
#    sudo apt install imx500-all
#    sudo apt install python3-opencv
#    sudo reboot
#
# 2) Build a venv, e.g.:
#    python3 -m venv /home/<user>/CAMERA/PYTHON/VirEnv --system-site-packages
#
# 3) Checkout the code.
#
# 4) Configure the shebang line containing <user> at the beginning of 'piedgecctv.py' to match your venv.
#
# 5) In 'conf/conf.py' configure 'base' and 'model' containing <user>.
#
# 6) Set your rpk 'model' of choice, info on models https://github.com/raspberrypi/imx500-models/blob/main/README.md
#
# 7) Ensure piedgecctv.py is executable:
#    chmod 755 piedgecctv.py
#
# 8) Operate from the venv:
#    source ~/CAMERA/PYTHON/VirEnv/bin/activate
#
# 9) Run:
#    ./piedgecctv.py


# Version 1.2.
# 
# PiEdgeCCTV is an edge AI CCTV system for Raspberry Pi using the IMX500 AI Camera.
# 
# It records video when a matched inference occurs providing combined pre-roll, event and post-roll capture.
#
# Version 1.2+ adds a web interface, local subnet only.
#
# Version 1.1+ includes optional motion detection which activates when light level is too low for matched inferences to occur.
# 
# Heavily based on https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py
# 
# See also the incredibly helpful Picamera2 Library manual https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
# 
# Plenty of room for improvement including post-processing framework, am not proud of the number of global variables.
# 
# Prototyped using a Raspberry Pi 5 16GB. IMX500 AI Camera info https://www.raspberrypi.com/documentation/accessories/ai-camera.html
#  
# Forum thread https://forums.raspberrypi.com/viewtopic.php?t=383752
#
# You might wish to increase the default Contiguous Memory Allocator (CMA) value for your Pi.
# 
# It is known 'Network Firmware Upload' sometimes fails (remains at 0%, sometimes repeating), CTRL-C and try again.


# Motion Detection (MD).
#
# Motion detection is an optional feature designed to operate at *night* when the camera lux value is below a threshold.
#
# It is motion detection between frames & does not use a PIR, based on https://github.com/raspberrypi/picamera2/blob/main/examples/capture_motion.py
#
# It can trigger for example when a security light has come on but the IMX500 offers very limited visual detail at low light so usefulness may vary.
#
# By default MD capability is disabled, to enable set 'mdenabled = True'.
#
# You will likely also need to determine values for 'mse_threshold' and 'metadata_lux_threshold' that work for your setup.
#
# MD can ignore the *top* part of the video 'ignore_hrows' to mask out swaying tree branches, clouds, etc, particularly useful during dawn or dusk. Recorded video is unchanged.
#
# MD operates on the 'lores' stream so timestamping is disabled to prevent interference.
#
# To show the preview window is live the current Lux value is displayed in the preview window title bar, this can also aid with tuning.
#
# At a glance it easy to see what configuration is in place:
#
# - Lux value shown in preview window title bar / no timestamp in preview window = AI and MD is enabled.
#
# - Timestamp show in preview window / no lux value in preview window title bar = only AI is enabled.
#
# In both cases a timestamp is applied to 'main' video recordings.


# Web interface - 'web.py'
#
# Inspired by Raspberry Pi Camera Guide 2nd Edition https://github.com/raspberrypipress/official-raspberry-pi-camera-guide-2e/tree/main/code/c
#
# The web interface automatically runs & by default is available at http://<ip address>:5000
#
# This will continue to run even when 'piedgecctv.py' is terminated, if you wish to stop it kill the process.


import datetime
import time
import os, sys
import shutil
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

### User configurable options have now moved to 'conf/conf.py' ###
import conf.conf as c


last_detections = []
tsfont = cv2.FONT_HERSHEY_SIMPLEX

base_output_folder = os.path.join(c.base, c.output_folder)
base_web_script = os.path.join(c.base, c.web_script)      
        
recordingai = c.recordingai
ltime = c.ltime
recordingmd = c.recordingmd
previous_frame = c.previous_frame


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
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=c.threshold, iou_thres=c.iou,
                                          max_out_dets=c.max_detections)[0]
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
        if score > c.threshold
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
    
        cv2.putText(m.array, ts, c.tsmorigin, tsfont, c.tsmscale, c.tscolour, c.tsmthickness)

    # Cannot have timestamp on lores as it interferes with motion detection
    if not c.mdenabled:
        with MappedArray(request, "lores") as n:
            cv2.putText(n.array, ts, c.tslorigin, tsfont, c.tslscale, c.tscolour, c.tslthickness)
        

def vidrecord():

    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    
    hit = False
           
    for detection in detections:
        label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
    
        lbl = labels[int(detection.category)]
        
        for i in c.interested:
            if i == lbl:
                hit = True
                break
    
    global recordingai
    global ltime
    global circular
    global ip, op, op_pre, op_pre_start_fix, op_post
    
    if not recordingai:    # recordingai = False
        now = datetime.datetime.now()
        nowform = now.strftime('%Y-%m-%d_%Hh%Mm%Ss')
        op_pre = os.path.join(c.mem_folder, nowform) + "_pre." + c.opformat
        op_pre_start_fix = os.path.join(c.mem_folder, nowform) + "_pre_start_fix." + c.opformat
        op_post = os.path.join(c.mem_folder, nowform) + "_post." + c.opformat
        ip = (op_pre_start_fix, op_post)
        os.makedirs(base_output_folder, exist_ok = True)
        if c.mdenabled:
            op = os.path.join(base_output_folder, nowform) + "_ai." + c.opformat
        else:
            op = os.path.join(base_output_folder, nowform) + "." + c.opformat
                   
    if hit and not recordingai:    # hit = True and recordingai = False, we need to start recording 
        print(f"Event (ai): {nowform}")
        circular.open_output(PyavOutput(f"{op_pre}", format=c.opformat))
        picam2.stop_encoder()
        output = PyavOutput(f"{op_post}", format=c.opformat)
        picam2.start_recording(encoder, output)
        recordingai = True
        ltime = time.time()
        
    elif hit and recordingai:    # hit = True and recordingai = True, make sure ltime is the last frame
        ltime = time.time()
        
    elif recordingai:    # recordingai = True, we need to stop recording
        if time.time() - ltime > c.postroll:
            picam2.stop_encoder()
            recordingai = False
            circular = CircularOutput2(buffer_duration_ms=c.preroll)
            picam2.start_recording(encoder, circular)
    
            # When events happen close together there might not be enough pre-roll available to make a file
            if os.path.exists(op_pre):
                vid_start_fix(op_pre, op_pre_start_fix)
                concatenate_vids(ip, op)
            else:
                shutil_return = shutil.move(op_post, op)
       

def vid_start_fix(faulty, fixed):
    
    # The flushed CircularOutput2 video has a start value which causes problems
    command = [
        "/usr/bin/ffmpeg",
        "-i", faulty,
        "-metadata", "start=0.000000",
        "-c", "copy",
        fixed,
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during vid_start_fix: {e}")
        print(f"ffmpeg output (stderr):\n{e.stderr}") #print ffmpeg error message.
    

def concatenate_vids(input_files, output_file):
    """Concatenates two or more video files using ffmpeg."""

    # Create a temporary file list for ffmpeg
    with open(c.text_file_shm, "w") as f:
        for file in input_files:
            f.write(f"file '{file}'\n")
        f.close()

    # Construct the ffmpeg command
    command = [
        "/usr/bin/ffmpeg",
        "-f", "concat",
        "-safe", "0",  # Allow unsafe file paths (if needed)
        "-i", c.text_file_shm,
        "-c", "copy",  # Copy streams without re-encoding
        output_file,
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during concatenate_vids: {e}")
        print(f"ffmpeg output (stderr):\n{e.stderr}") #print ffmpeg error message.
    finally:
        # Clean up temporary files
        os.remove(c.text_file_shm)
        os.remove(op_pre)
        for z in input_files:
            os.remove(z)


def motiondetect():
    
    global recordingmd
    global ltime
    global circular
    global ip, op, op_pre, op_pre_start_fix, op_post
    global previous_frame

    current_frame = picam2.capture_buffer("lores")

    # Catch when 'c.ignore_hrows = 0'
    if c.ignore_hrows == 0:
        current_frame = current_frame[:c.loresw * c.loresh].reshape(c.loresh, c.loresw)
    else:
        current_frame = current_frame[c.loresw * c.ignore_hrows:c.loresw * c.loresh].reshape((c.loresh - c.ignore_hrows), c.loresw)
    
    if previous_frame is not None:
        # Measure pixels differences between current and previous frame
        mse = np.square(np.subtract(current_frame, previous_frame)).mean()

        if mse > c.mse_threshold:
            if not recordingmd:
                now = datetime.datetime.now()
                nowform = now.strftime('%Y-%m-%d_%Hh%Mm%Ss')
                op_pre = os.path.join(c.mem_folder, nowform) + "motion_pre." + c.opformat
                op_pre_start_fix = os.path.join(c.mem_folder, nowform) + "motion_pre_start_fix." + c.opformat
                op_post = os.path.join(c.mem_folder, nowform) + "motion_post." + c.opformat
                ip = (op_pre_start_fix, op_post)
                os.makedirs(base_output_folder, exist_ok = True)
                op = os.path.join(base_output_folder, nowform) + "_md." + c.opformat

                print(f"Event (md): {nowform}  (mse: {mse:.2f})")
                
                circular.open_output(PyavOutput(f"{op_pre}", format=c.opformat))
                picam2.stop_encoder()
                output = PyavOutput(f"{op_post}", format=c.opformat)
                picam2.start_recording(encoder, output)

                recordingmd = True
            ltime = time.time()
        else:
             if recordingmd and time.time() - ltime > c.postroll:
                picam2.stop_encoder()
                recordingmd = False

                circular = CircularOutput2(buffer_duration_ms=c.preroll)
                picam2.start_recording(encoder, circular)
                
                # When events happen close together there might not be enough pre-roll available to make a file
                if os.path.exists(op_pre):
                    vid_start_fix(op_pre, op_pre_start_fix)
                    concatenate_vids(ip, op)
                else:
                    shutil_return = shutil.move(op_post, op)
                                           
    previous_frame = current_frame


if __name__ == "__main__":

    imx500 = IMX500(c.model)
    intrinsics = imx500.network_intrinsics
    
    picam2 = Picamera2(imx500.camera_num)
    
    main  = {'size': (c.mainw, c.mainh), 'format': 'XRGB8888'}
    if c.mdenabled:
        lores = {'size': (c.loresw, c.loresh), 'format': 'YUV420'}  # MD works best on Y (we ignore U & V)
    else:
        lores = {'size': (c.loresw, c.loresh), 'format': 'XRGB8888'}  # Invalid for c.models lower than RPi5

    camcontrols = {'FrameRate': c.fps}
    
    # Might turn the camera upside down later so leaving in transform
    config = picam2.create_video_configuration(main, lores=lores, controls=camcontrols, buffer_count=12, encode="main", display="lores", transform=Transform(hflip=False, vflip=False))

    encoder = H264Encoder(bitrate=10000000)

    imx500.show_network_fw_progress_bar()
    
    picam2.pre_callback = apply_timestamp

    #picam2.start_preview(Preview.QT, x=2, y=64)
    picam2.start_preview(Preview.QTGL, x=2, y=64)
    
    if c.mdenabled:
        # Display Lux in preview window title bar to show the preview window is actively 'live', also useful for tuning.
        picam2.title_fields = ["Lux"]

    picam2.start(config, show_preview=True)
    
    circular = CircularOutput2(buffer_duration_ms=c.preroll)

    picam2.start_recording(encoder, circular)
    
    last_results = None
    
    
    # Launch the web interface as a separate process only if not currently running
    spcommand = (f"/usr/bin/ps aux | /usr/bin/grep {c.web_script} | /usr/bin/grep -v grep")
    ps = subprocess.run(
        [spcommand], shell=True, capture_output=True, text=True
        )
    if ps.returncode != 0:
        try:
            process = subprocess.Popen(
                [sys.executable, base_web_script],       
                #preexec_fn=os.setsid,
                start_new_session=True,  # Creates a new session for the child process
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"Child script '{c.web_script}' spawned with PID: {process.pid}")
        except Exception as e:
            print(f"Error spawning child process: {e}")
    else:
        print(f"Child script '{c.web_script}' is already running")


    while True:
 
        if c.mdenabled:

            # Only check lux value at intervals
            if  time.time() - c.cltime > c.check_lux_every:           
                metadata = picam2.capture_metadata()
                c.cltime = time.time()
                if metadata['Lux'] < c.metadata_lux_threshold:
                    c.nightime = True
                else:
                    c.nightime = False
             
            # Day to night
            if c.nightime:
                # Don't move from AI to MD until an existing AI recording completes
                if recordingai:
                    last_results = parse_detections(picam2.capture_metadata())
                    vidrecord() 
                else:
                    motiondetect()
            
            # Night to day
            else:
                # Don't move from MD to AI until an existing MD recording completes
                if recordingmd:
                    motiondetect()
                else:
                    last_results = parse_detections(picam2.capture_metadata())
                    vidrecord()
                    
        else:
                        
            last_results = parse_detections(picam2.capture_metadata())
            vidrecord()         
  
  
