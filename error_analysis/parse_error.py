import json
filename = "error_output_small_for_parse.json"
image_ids = open("image_ids.txt", "w+")
image_id_list = []
with open(filename) as file:
	data = json.load(file)['result']
	for i in data:
		image_ids.write(i['image_id']+"\n")
		image_id_list.append(i['image_id'])

#loop through images in visualDialog_val
import os
from shutil import copyfile

for filename in os.listdir("VisualDialog_val2018"):
    if filename.endswith(".jpg"): 
    	name = filename.split(".jpg")[0]
    	for image_id in image_id_list:
    		if name.endswith(str(image_id)):
    			if name.split(str(image_id))[0].endswith("0"):
    				copyfile("./VisualDialog_val2018/"+filename, "./error_output_images_small/"+filename)