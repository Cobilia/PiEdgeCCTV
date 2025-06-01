# PiEdgeCCTV
## PiEdgeCCTV is an edge AI CCTV system for Raspberry Pi using the IMX500 AI Camera

It records video when a matched inference occurs providing combined pre-roll, event and post-roll capture.

Version 1.2+ adds a web interface, local subnet only. 

![Image](https://github.com/user-attachments/assets/fe305560-af68-4633-8b01-0fe84f533be2)

Version 1.1+ includes optional motion detection which activates when light level is too low for matched inferences to occur.

Heavily based on https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py

See also the incredibly helpful Picamera2 Library manual https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

Plenty of room for improvement including post-processing framework, am not proud of the number of global variables.

Prototyped using a Raspberry Pi 5 16GB. IMX500 AI Camera info https://www.raspberrypi.com/documentation/accessories/ai-camera.html

Tested on Raspberry Pi OS version Debian 12.11.
 
Forum thread https://forums.raspberrypi.com/viewtopic.php?t=383752

You might wish to increase the default Contiguous Memory Allocator (CMA) value for your Pi.

It is known 'Network Firmware Upload' sometimes fails (remains at 0%, sometimes repeating), CTRL-C and try again.  


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


### Web interface - 'web.py'

Inspired by Raspberry Pi Camera Guide 2nd Edition https://github.com/raspberrypipress/official-raspberry-pi-camera-guide-2e/blob/main/code/ch16/camera_underwater.py

The web interface automatically runs & by default is available at http://\<ip address\>:5000

This will continue to run even when 'piedgecctv.py' is terminated, if you wish to stop it kill the process.  

"Proc running:" shows if the 'piedgecctv.py' process is running.


### Installation

1) sudo apt update && sudo apt full-upgrade  
   sudo apt install imx500-all  
   sudo apt install python3-opencv  
   sudo reboot  

2) Build a venv:  
   python3 -m venv /home/\<user\>/CAMERA/PYTHON/VirEnv --system-site-packages

3) Check out the latest release.

4) Configure the shebang line containing \<user\> at the beginning of 'piedgecctv.py' to match your venv.

5) In 'conf/conf.py' configure 'base' and 'model' containing \<user\>.

6) Set your rpk 'model' of choice, info on models https://github.com/raspberrypi/imx500-models/blob/main/README.md

7) Ensure 'piedgecctv.py' is executable:  
   chmod 755 piedgecctv.py

8) Operate from the venv:  
   source ~/CAMERA/PYTHON/VirEnv/bin/activate

9) Run:  
   ./piedgecctv.py  

### Structure

![Image](https://github.com/user-attachments/assets/2646315c-17bb-4bc4-a938-79c9176605ba)


  
