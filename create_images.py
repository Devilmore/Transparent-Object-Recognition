from random import *
from moviepy.editor import *
from PIL import *
import os.path
import time
from string import digits

video_path = "../videos/"

# Iterate over all video files in the given path
for j in os.listdir(video_path):
    if j.endswith(".ogv"): 
        video = VideoFileClip(video_path + str(j))
	# Get the filename without numbers or extension and make a subfolder of it to contain all images with that label
	filename = j.replace(".ogv","").translate(None, digits)
	try: 
		os.mkdir(video_path + filename)
	except OSError:
		# Jump ahead somewhere to get better randint results. Might as well be here since except wanted to do something
		jumpahead(20)
	for x in range(0, 10):
		# Choose a random second
		i = randint(0,int(video.duration))
		# Save the image (it would be better if this step was unnecessary but moviepy is being a bitch about it)
		output_str = video_path + filename + "/" + filename + str(i) + '.jpg'
		video.save_frame(output_str, i)
		# Open the image again, choose the needed area (this has to be changed or ommitted for different setups) and save it again, overwriting the first file
		image = Image.open(output_str)
		box = (20,40,490,510)
		region = image.crop(box)
		region.save(output_str)
    else:
        continue


