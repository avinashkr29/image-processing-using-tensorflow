
import tensorflow as tf
import numpy as np
import os
from PIL import Image

cur_dir = os.getcwd()
print("resizing images")
print("current directory:",cur_dir)

filenames=[]
for i in os.listdir():
    filenames.append(i)
    
def inputs():
    filename_queue = tf.train.string_input_producer(filenames)    
    filename,value = tf.WholeFileReader().read(filename_queue)
    image = tf.image.decode_jpeg(value)
    resized = tf.image.resize_images(image, (64, 64), 1)
    #resized.set_shape([180,180,3])
    #reshaped_image= tf.image.flip_up_down(resized) 
    return filename,resized

with tf.Graph().as_default():
    image = inputs()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    for i in filenames:
        filename,img = sess.run(image)
        img = Image.fromarray(img, "RGB")
        img.save(os.path.join(r'path_of_directory',filename.decode("utf-8")))
