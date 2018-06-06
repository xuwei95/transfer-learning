import os
import getfeature
import tensorflow as tf
sess=tf.Session()
def create_bottleneck(path,savepath):
    label_lists = os.listdir(path)
    for sub_dir in label_lists:
        sub_dir_path = os.path.join(savepath, sub_dir)
        if not os.path.exists(sub_dir_path):
            os.makedirs(sub_dir_path)
        for i in os.listdir('%s/%s'%(path,sub_dir)):
            img = '%s/%s/%s' % (path, sub_dir, i)
            bottleneck_values = getfeature.getbottleneck_values(sess,img)
            getfeature.save(bottleneck_values, '%s/%s/%s.txt' % (savepath, sub_dir, i))
create_bottleneck('test','1')