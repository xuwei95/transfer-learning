import tensorflow as tf
import input_data
import sys
n_classes=12
train_x, train_y, val_x, val_y= input_data.get_files('features',0.1)
test_x,test_y=input_data.gettest(val_x, val_y)
batch_size = 100
num_batch = len(train_x) // batch_size
maxstep=5000
x = tf.placeholder(tf.float32, [None, 2048])
y_ = tf.placeholder(tf.float32, [None, n_classes])
def nnLayer():
    # 输出层
    W = tf.get_variable('softmax_linear',
                         shape=[2048, n_classes],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.0, dtype=tf.float32))
    b = tf.get_variable('biases',
                         shape=[n_classes],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0.1))
    out = tf.add(tf.matmul(x, W), b)
    return out

def Train():
    out = nnLayer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        for n in range(10000):
             # 每次取batch_size张图片
            for i in range(num_batch):
                batch_x,batch_y=input_data.getbatch(train_x,train_y,i,batch_size)
                # 开始训练数据，同时训练三个变量，返回三个数据
                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y})
                summary_writer.add_summary(summary, n*num_batch+i)
                #打印损失
                print(n*num_batch+i, loss)

                if (n*num_batch+i) % 100 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x:test_x, y_:test_y})
                    print('step:',n*num_batch+i, 'loss:',loss,'acc:',acc)
                    # 准确率大于0.9
                if acc > 0.90 or n * num_batch + i >= maxstep:
                    saver.save(sess, './model/train.model', global_step=n * num_batch + i)
                    sys.exit(0)

Train()