# PiEdgeCCTV
PiEdgeCCTV is an edge AI CCTV system for Raspberry Pi using the IMX500 AI Camera.

It records video when a matched inference occurs providing combined pre-roll, event and post-roll capture.

Heavily based on https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py

See also the incredibly helpful Picamera2 Library manual https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

Plenty of room for improvement including post-processing framework, am not proud of the number of global variables.

Prototyped using a Raspberry Pi 5 16GB. IMX500 AI Camera info https://www.raspberrypi.com/documentation/accessories/ai-camera.html

Info on models https://github.com/raspberrypi/imx500-models/blob/main/README.md

Forum thread https://forums.raspberrypi.com/viewtopic.php?t=383752

You might wish to increase the default Contiguous Memory Allocator (CMA) value for your Pi, see Picamera2 Library manual.

It is known 'Network Firmware Upload' sometimes fails, just try again.
