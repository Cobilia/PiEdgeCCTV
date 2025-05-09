# PiEdgeCCTV
## PiEdgeCCTV is an edge AI CCTV system for Raspberry Pi using the IMX500 AI Camera

It records video when a matched inference occurs providing combined pre-roll, event and post-roll capture.

Version 1.1+ includes optional motion detection which activates when light level is too low for matched inferences to occur.

Heavily based on https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py

See also the incredibly helpful Picamera2 Library manual https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

Plenty of room for improvement including post-processing framework, am not proud of the number of global variables.

Prototyped using a Raspberry Pi 5 16GB. IMX500 AI Camera info https://www.raspberrypi.com/documentation/accessories/ai-camera.html

Info on models https://github.com/raspberrypi/imx500-models/blob/main/README.md

Forum thread https://forums.raspberrypi.com/viewtopic.php?t=383752

You might wish to increase the default Contiguous Memory Allocator (CMA) value for your Pi.

It is known 'Network Firmware Upload' sometimes fails, just try again.




### Motion Detection (MD)

Motion detection is an optional feature designed to operate at *night* when the camera lux value is below a threshold.

It is motion detection between frames & does not use a PIR, based on https://github.com/raspberrypi/picamera2/blob/main/examples/capture_motion.py

It can trigger for example when a security light has come on but the IMX500 offers very limited visual detail at low light so usefulness may vary.

By default MD capability is disabled, to enable set '**mdenabled = True**'.

You will likely also need to determine values for '**mse_threshold**' and '**metadata_lux_threshold**' that work for your setup.

MD can ignore the *top* part of the video '**ignore_hrows**' to mask out swaying tree branches, clouds, etc, particularly useful during dawn or dusk. Recorded video is unchanged.

MD operates on the 'lores' stream so timestamping is disabled to prevent interference.

To show the preview window is live the current Lux value is displayed in the preview window title bar, this can also aid with tuning.

At a glance it easy to see what configuration is in place:

- Lux value shown in preview window title bar / no timestamp in preview window = AI and MD is enabled.

- Timestamp show in preview window / no lux value in preview window title bar = only AI is enabled.

In both cases a timestamp is applied to 'main' video recordings.

Keep an eye on the contents of '**mem_folder**' for any failed recordings, though hopefully this shouldn't happen.


