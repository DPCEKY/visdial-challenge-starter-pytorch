import sys
from PIL import Image

image_id = sys.argv[1]
image_name = "./error_output_images_small/VisualDialog_val2018_"+"0"*(12-len(image_id))+str(image_id)+".jpg"
im = Image.open(image_name)
im.show()

