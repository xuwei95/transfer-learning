import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
import os

# 在训练模型时，可以通过tensor.name来获取张量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# 图像输入张量所对应的名称。
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
# 读取已经训练好的Inception-v3模型。
with gfile.FastGFile(os.path.join('model/tensorflow_inception_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
# 加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量。
bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                      JPEG_DATA_TENSOR_NAME])
def getbottleneck_values(sess,image_path):
    # 获取图片内容。
    image_data = gfile.FastGFile(image_path, 'rb').read()
    # 这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值就是这张图片的新的特征向量。
    bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values
#将特征向量保存到文件
def save(bottleneck_values,file):
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(file, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)
#读取特征向量文件
def read(file):
    with open(file, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    bottleneck_values=np.array(bottleneck_values)
    return bottleneck_values