
# Code base directory
base = "/home/<user>/CAMERA/PYTHON/"

# Final destination for video files
output_folder = "static/vids"

# Memory location
mem_folder = "/dev/shm"

# ffmpeg concat text file
text_file_shm = '/dev/shm/input_list.txt'

# When starting, 'False' means it does not begin recording AI straight away
recordingai = False

# When starting, 'False' means it does not begin recording MD straight away
recordingmd = False

# Initial time difference set to zero at start
ltime = 0

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
#tsfont = cv2.FONT_HERSHEY_SIMPLEX
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



#### AI ####

model = '/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk'
threshold = 0.65
iou = 0.65
max_detections = 5

# What we want to record
interested = ('person', 'cat', 'dog') # NOTE: if there is only one item in this list there MUST be a trailing comma e.g. ('dog',)!



#### MOTION DETECTION (MD) ####

# Enable or disable motion detection
mdenabled = False
# Mean Square Error (think of this as the difference between frames), only record when the mse is above this value
mse_threshold = 2.9
# Switch to nocturnal MD when the lux level falls below this value
metadata_lux_threshold = 3.0
# Only check lux every (s), 300 is a compromise between too low causing flapping & too long missing a change
check_lux_every = 300 
# Check lux initial time difference set to zero at start
cltime = 0
# Set 'previous_frame' to None to start with
previous_frame = None
# If False do not start in nightime (MD) mode
nightime = False
# Ignores this number of height rows starting from the *top*, useful for filter out swaying tree branches & moving clouds during dawn or dusk
ignore_hrows = 0  # e.g. 0 provides full height MD, 400 means only 80 rows at the *bottom* detect motion (for a 480 'loresh' value)



#### Web Interface ####

main_process = "piedgecctv.py"
web_port = 5000
web_script = 'web.py'

