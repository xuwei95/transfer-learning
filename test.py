import getfeature
import tensorflow as tf
import os
import numpy as np
n_classes=12
x = tf.placeholder(tf.float32, [None, 2048])
y_ = tf.placeholder(tf.float32, [None, n_classes])
def nnLayer():
    # 输出层
    W = tf.get_variable('softmax_linear',
                         shape=[2048, n_classes],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.0, dtype=tf.float32))
    b = tf.get_variable('biases',
                         shape=[12],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0.1))
    out = tf.add(tf.matmul(x, W), b)
    return out
output = nnLayer()
predict = tf.argmax(output, 1)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('./model/'))
def testimg(img):
    s=getfeature.getbottleneck_values(sess,img)
    s=[s]
    s=np.array(s)
    res = sess.run(predict, feed_dict={x:s})
    return res
def test(path):
    list=os.listdir(path)
    a=0
    b=0
    labels = os.listdir('train')
    for i in list:
        for j in os.listdir("%s/%s"%(path,i)):
            img="%s/%s/%s"%(path,i,j)
            lb=testimg(img)
            tlb=labels.index(i)
            if lb==tlb:
                b+=1
            a+=1
    print(b/a)
test("test")
# a=testimg('test.jpg')
# print(a)