import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


def add_layer(inputs, in_size, out_size, n_layer, keep_prob=1, activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer  ## define a new var
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),
                                  name='W')  # 权值，tf.random_normal([n, m])代表生成一个 n x m 矩阵
            # tf.histogram_summary(layer_name+'/weights',Weights)   # tensorflow 0.12 以下版的
            tf.summary.histogram('/value', Weights)  # tensorflow >= 0.12
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')  # 偏移，tf.zeros([n, m])代表生成一个 n x m 矩阵
            # tf.histogram_summary(layer_name+'/biase',biases)   # tensorflow 0.12 以下版的
            tf.summary.histogram('/value', biases)  # Tensorflow >= 0.12
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)  # y = wx + b
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)  # 利用dropout防止过拟合
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        # tf.histogram_summary(layer_name+'/outputs',outputs) # tensorflow 0.12 以下版本
        tf.summary.histogram('/outputs', outputs)  # Tensorflow >= 0.12

        return outputs


# 训练函数拟合
def test1():
    # 构建数据
    x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)  # 噪音
    y_data = np.square(x_data) - 0.5 + noise

    # 占位符，这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1
    with tf.name_scope('inputs'):
        # define placeholder for inputs to network
        xs = tf.placeholder(tf.float32, [None, 1])
        ys = tf.placeholder(tf.float32, [None, 1])

    # 定义隐藏层，这里使用Tensorflow 自带的激励函数tf.nn.relu
    L1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
    # 定义输出层
    prediction = add_layer(L1, 10, 1, n_layer=2, activation_function=None)
    #   计算预测值prediction和真实值的误差，对二者差的平方求和再取平均
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
        # tf.scalar_summary('loss',loss) # tensorflow < 0.12
        tf.summary.scalar('value', loss)  # tensorflow >= 0.12，scalar的是放在SCALARS下面

    # 接下来，是很关键的一步，如何让机器学习提升它的准确率。tf.train.GradientDescentOptimizer()中的值通常都小于1，这里取的是0.1，代表以0.1的效率来最小化误差loss
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 选择梯度下降算法来学习

    # 定义Session，并用 Session 来执行 init 初始化步骤。 （注意：在tensorflow中，只有session.run()才会执行我们定义的运算）
    sess = tf.Session()  # get session
    # merged= tf.merge_all_summaries()    # tensorflow < 0.12
    merged = tf.summary.merge_all()  # tensorflow >= 0.12，把所有的 summaries 合并到一起
    # writer = tf.train.SummaryWriter('logs/', sess.graph)    # tensorflow < 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)  # tensorflow >=0.12

    # 使用变量时，都要对它进行初始化，这是必不可少的
    # sess.run(tf.initialize_all_variables()) # tf.initialize_all_variables() # tf 马上就要废弃这种写法
    sess.run(tf.global_variables_initializer())  # 替换成这样就好

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()  # 开启交互模式，这种模式下plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()，但是图像不会停留
    plt.show()

    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  # train_step相当于表达式，返回值相当于执行表达式的输出值
        if i % 50 == 0:
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])  # 删掉之前的线，重新绘制
            except Exception:
                pass

            # 每隔上50次就记录一次结果
            rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(rs, i)

            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', linewidth=5)  # r-代表红色粗实线
            plt.pause(0.1)

    plt.ioff()  # 关闭交互模式，图像可以停留
    plt.show()


# 训练图片识别
def test2():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # MNIST库是手写体数字库,数据中包含55000张训练图片，每张图片的分辨率是28×28
    xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])  # 每张图片都表示一个数字，所以我们的输出是数字0到9，共10类
    prediction = add_layer(xs, 784, 10, n_layer=1, activation_function=tf.nn.softmax)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss，tf.log是取对数
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.Session()
    # tf.initialize_all_variables() 这种写法马上就要被废弃
    # 替换成下面的写法:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次只取100张图片，免得数据太多训练太慢
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 50 == 0:
            print(compute_accuracy(sess, xs, ys, prediction, mnist.test.images,
                                   mnist.test.labels))  # mnist提供的测试图片数据，有10000个


def compute_accuracy(sess, xs, ys, prediction, v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})

    # tf.argmax参数v_ys代表一个 batch_size * 10 的二维数组，参数1代表求行的最大值，0代表求列的最大值，也就是说，只在每一行选取最大值对应的下标。
    # 于是得到的结果是一个长度为 batch 的一维数组，这个一维数组中的值就表示了每一个样例对应的数字识别结果
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))

    # 这个运算首先将一个布尔型的数值转换为实数型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# 测试防止过拟合效果
def test3():
    # load data
    digits = load_digits()
    x = digits.data
    y = digits.target
    y = LabelBinarizer().fit_transform(y)  # 将y转换成batch_size * 10 的二维数组
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)  # 其中X_train是训练数据, X_test是测试数据

    # define placeholder for inputs to network
    keep_prob = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
    ys = tf.placeholder(tf.float32, [None, 10])

    # add output layer
    L1 = add_layer(xs, 64, 50, 'l1', keep_prob=keep_prob, activation_function=tf.nn.tanh)
    prediction = add_layer(L1, 50, 10, 'l2', activation_function=tf.nn.softmax)

    # the loss between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.Session()
    merged = tf.summary.merge_all()
    # summary writer goes in here
    train_writer = tf.summary.FileWriter("logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test", sess.graph)

    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(500):
        # here to determine the keeping probability
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
        if i % 50 == 0:
            # record loss
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            train_writer.add_summary(train_result, i)  # train_result代表y轴值，i代表x轴值
            test_writer.add_summary(test_result, i)


# 卷积神经网络
def test4():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # 把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，
    # 因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3
    x_image = tf.reshape(xs, [-1, 28, 28, 1])

    ## 第一个卷积层 ##
    W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积核patch的大小是5x5,相当于扫描区域，因为黑白图片channel是1所以输入是1，输出是32个featuremap
    b_conv1 = bias_variable([32])  # 定义bias，它的大小是32个长度，因此我们传入它的shape为[32]
    h_conv1 = tf.nn.relu(conv2d(x_image,
                                W_conv1) + b_conv1)  # 定义卷积神经网络的第一个卷积层，因为采用了SAME的padding方式，输出图片的大小没有变化依然是28x28，output size 28x28x32
    h_pool1 = max_pool_2x2(h_conv1)  # 定义pooling，压缩后大小为 14x14x32

    ## 第二个卷积层 ##
    W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64，32代表in channel，即卷积核的深度，这样扫描区域就是 5 x 5 x 32
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

    ## 连接层，负责将三维数据变成一维数据 ##
    # weight_variable的shape输入就是第二个卷积层展平了的输出大小: 7x7x64， 后面的输出size我们继续扩大，定为1024，
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    # 通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出结果展平
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # 计算输出值
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # 筛选结果，防止过拟合
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## fc2 layer ##
    # 最后的输出是10个 (因为mnist数据集就是[0-9]十个类)，prediction就是我们最后的预测值
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 利用交叉熵损失函数来定义我们的cost function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.Session()
    # important step
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    if os.path.exists('my_net'):
        saver.restore(sess, "my_net/save_net.ckpt")

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            print(compute_accuracy2(sess, xs, ys, prediction, keep_prob,
                                    mnist.test.images[:1000], mnist.test.labels[:1000]))

    save_path = saver.save(sess, "my_net/save_net.ckpt")  # 保存训练结果
    print("save result to %s" % (save_path))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # 定义卷积，tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，x是图片的所有参数，W是此卷积层的权重，
    # 然后定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，
    # 中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME
    # W是CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape
    # SAME表示卷积核左边界可以停留在图像边缘，这样生成的特征层大小和原数据大小完全一样，如果使用VALID代表卷积核右边界可以停留在图像边缘
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 定义池化pooling，卷积层的padding我们选的是一次一步，也就是strides[1]=strides[2]=1，这样得到的图片尺寸没有变化，
    # 而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，pooling 有两种，一种是最大值池化，最大池化相当于选择范围内最大的特征值作为结果
    # 一种是平均值池化，平均池化是选择范围内所有特征值的平均值作为结果，本例采用的是最大值池化tf.max_pool()。
    # 池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]，相当于图片长宽都压缩为原来的一半
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def compute_accuracy2(sess, xs, ys, prediction, keep_prob, v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


if __name__ == '__main__':
    test4()
