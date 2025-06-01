
# Inspired by Raspberry Pi Camera Guide 2nd Edition https://github.com/raspberrypipress/official-raspberry-pi-camera-guide-2e/blob/main/code/ch16/camera_underwater.py

from flask import Flask, render_template, request, redirect, url_for
import os, psutil
import datetime

# User configurable variables
import conf.conf as c

base_output_folder = os.path.join(c.base, c.output_folder)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/', methods = ['POST','GET'])
def page():

    if request.method == 'POST':

        if request.form['submit'] == 'Delete':
            cbx = request.form.getlist('cbox')
            
            if len(cbx) > 0:
                for cb in cbx:
                    cbpath = os.path.join(base_output_folder, cb)
                    if os.path.exists(cbpath):
                        os.remove(cbpath)
                        
            # Post/Redirect/Get (PRG) Design Pattern
            return redirect(url_for('page'))
 
    df_root = os.statvfs('/') # are we running out of root disk space?
    df_root_size = df_root.f_frsize * df_root.f_blocks
    df_root_avail = df_root.f_frsize * df_root.f_bfree
    df_root_pc = round(100 * df_root_avail/df_root_size)
    
    df_memdisk = os.statvfs(c.mem_folder) # are we running out of mem disk space?
    df_memdisk_size = df_memdisk.f_frsize * df_memdisk.f_blocks
    df_memdisk_avail = df_memdisk.f_frsize * df_memdisk.f_bfree
    df_memdisk_pc = round(100 * df_memdisk_avail/df_memdisk_size)
   
    
    vids = os.listdir(base_output_folder)
    vids.sort(reverse = True)
    
    proc = False
    for p in psutil.process_iter(['pid', 'name']):
        if p.info['name'] == c.main_process:
            if not p.status() == psutil.STATUS_ZOMBIE:
                proc = True
                
    now = datetime.datetime.now()
    nowform = now.strftime('%d %b %Y @ %H:%M')
            

    # Display the web page template with our template variables
    return render_template('index.html', opf=c.output_folder, nowform=nowform, df_root_pc=df_root_pc, df_memdisk_pc=df_memdisk_pc, proc=proc, vids=vids)

if __name__ == "__main__":

    app.run(host='0.0.0.0',port=c.web_port,debug=False)
