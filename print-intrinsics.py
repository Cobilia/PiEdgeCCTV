#!/home/<user>/CAMERA/PYTHON/VirEnv/bin/python


# Prints intrinsics classes labels for a specific rpk file.
# 
# Required labels can then be added to 'interested' in piedgecctv.py
#
# This is a shortcut script for functionality available in https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py
#
# The original above also provides acces to 'assets/coco_labels.txt' whereas this doesn't.


# 0) Configure lines containing <user> with settings for your environment.
#
# 1) sudo apt update && sudo apt full-upgrade
#    sudo apt install imx500-all
#    sudo reboot
#
# 2) Build a venv:
#    python3 -m venv /home/<user>/CAMERA/PYTHON/VirEnv --system-site-packages
#
# 3) Operate from the venv:
#    source ~/CAMERA/PYTHON/VirEnv/bin/activate
#
# 4) Ensure print-intrinsics.py is executable:
#    chmod 755 print-intrinsics.py
#
# 5) Run:
#    ./print-intrinsics.py <full-path-to-the-rpk-file>
#
# 6) Example:
#    ./print-intrinsics.py /usr/share/imx500-models/imx500_network_efficientdet_lite0_pp.rpk
#    Produces:
#    ..."classes": {"labels": ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"...


import sys

# [0] is script name. [1] is the rpk argument.
if len(sys.argv) != 2:
    print("\n Requires one argument, path to the rpk e.g.:")
    print("", sys.argv[0], "/usr/share/imx500-models/imx500_network_efficientdet_lite0_pp.rpk\n")
    exit()
    
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics)

imx500 = IMX500(sys.argv[1])
intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "object detection"
elif intrinsics.task != "object detection":
    print("Network is not an object detection task\n", file=sys.stderr)
    exit()

print(intrinsics, "\n")
