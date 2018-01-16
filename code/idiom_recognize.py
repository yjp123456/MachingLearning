# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

template = [[1.0, 0.0], [0.0, 1.0]]


def add_layer(inputs, in_size, out_size, n_layer, keep_prob=1, activation_function=None, norm=False):
    # 数据预处理，让数据具备统一规格
    if norm:
        # Batch Normalize
        fc_mean, fc_var = tf.nn.moments(
            inputs,
            axes=[0],  # the dimension you wanna normalize, here [0] for batch
            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
        )
        scale = tf.Variable(tf.ones([in_size]))
        shift = tf.Variable(tf.zeros([in_size]))
        epsilon = 0.001

        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = mean_var_with_update()

        inputs = tf.nn.batch_normalization(inputs, mean, var, shift, scale, epsilon)
        # similar with this two steps:
        # inputs = (inputs - fc_mean) / tf.sqrt(fc_var + 0.001)
        # inputs = inputs * scale + shift

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


def loadData(data):
    # 因为输入是float类型，所以需要把汉字转成byte，再把byte转成float
    result = []
    i = 0
    for item in data:
        result.append([])
        item = bytearray(item, "utf-8")
        for b in item:
            b = float(b)
            result[i].append(b)
        i += 1
    return result


def run(size):
    with tf.name_scope('inputs'):
        # define placeholder for inputs to network
        xs = tf.placeholder(tf.float32, [None, 12])
        ys = tf.placeholder(tf.float32, [None, 2])

    # 定义隐藏层，这里使用Tensorflow 自带的激励函数tf.nn.relu
    L1 = add_layer(xs, 12, size, n_layer=1, activation_function=tf.nn.relu, norm=True)  # 定义输出层
    prediction = add_layer(L1, size, 2, n_layer=2, activation_function=None, norm=True)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))

    # 这个运算首先将一个布尔型的数值转换为实数型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #   计算预测值prediction和真实值的误差，对二者差的平方求和再取平均
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss，tf.log是取对数
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
    total = 0
    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  # train_step相当于表达式，返回值相当于执行表达式的输出值
        if i % 50 == 0:
            # to visualize the result and improvement
            # 每隔上50次就记录一次结果
            # rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            # writer.add_summary(rs, i)

            prediction_value = sess.run(prediction, feed_dict={xs: test_data_x, ys: test_data_y})
            accuracy_value = sess.run(accuracy, feed_dict={xs: test_data_x, ys: test_data_y})
            print("预测值：%s" % (prediction_value))
            print("准确度:%s" % (accuracy_value))
            total += accuracy_value
            # plot the prediction
    return total / 20


# 卷积神经网络
def juanji(size):
    xs = tf.placeholder(tf.float32, [None, 12])
    ys = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)

    # 把xs的形状变成[-1,1,12,1]，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，
    # 因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3
    x_image = tf.reshape(xs, [-1, 1, 12, 1])  # 转换成一个四维数组，其中最外层行数自动计算

    ## 第一个卷积层 ##
    W_conv1 = weight_variable([1, 3, 1, size])  # 卷积核patch的大小是1x3,相当于扫描区域，因为黑白图片channel是1所以输入是1，输出是size个featuremap
    b_conv1 = bias_variable([size])
    h_conv1 = tf.nn.relu(conv2d(x_image,
                                W_conv1) + b_conv1)
    h_pool1 = max_pool_1x3(h_conv1)  # 定义pooling，压缩后大小为 1x4xsize

    ## 第二个卷积层 ##
    W_conv2 = weight_variable([1, 2, size, 2 * size])
    b_conv2 = bias_variable([2 * size])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_1x2(h_conv2)  # output size 1x2x2xsize

    ## 连接层，负责将三维数据变成一维数据 ##
    W_fc1 = weight_variable([1 * 2 * 2 * size, 102])
    b_fc1 = bias_variable([102])

    # 通过tf.reshape()将h_pool2的输出值从一个三维的变为二维的数据, -1表示自动计算行数,下面的结果只有一行，所以相当于将上一个输出结果展平
    h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 2 * 2 * size])
    # 计算输出值
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # 筛选结果，防止过拟合
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## fc2 layer ##
    # 最后的输出是2个，prediction就是我们最后的预测值
    W_fc2 = weight_variable([102, 2])
    b_fc2 = bias_variable([2])
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

    total = 0
    for i in range(1000):
        batch_xs, batch_ys = x_data, y_data
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1.0})
        if i % 50 == 0:
            result = compute_accuracy2(sess, xs, ys, prediction, keep_prob,
                                       test_data_x, test_data_y)
            print("result is:%f" % (result))
            total += result
    return total / 20


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x3(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME')


def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')


def compute_accuracy2(sess, xs, ys, prediction, keep_prob, v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    print("预测值:%s" % (y_pre))
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def readIdiom():
    global template
    result_x = []
    result_y = []
    file_object = open('idiom', 'r', encoding='utf-8')
    try:
        lines = file_object.readlines()
        for line in lines:
            if line:
                str = line.strip().split(" ")
                result_x.append(str[0])
                result_y.append(template[int(str[1])])
    finally:
        file_object.close()
    print(result_x)
    print(result_y)
    return result_x, result_y


# 识别输入的四个汉字是不是成语，这边是自己定义数据，最好数据源是来自汉语词典
x_data, y_data = readIdiom()
x_data = np.array(loadData(x_data))
x_data = x_data.reshape([-1, 12])
y_data = np.array(y_data)
y_data = y_data.reshape([-1, 2])
hidden_x = []
hidden_y = []
n_hidden_size = 6

data_test = ["千方百计", "测试测试", "一暴十日", "一五一十", "千篇一律", "万无一失", "一五一六", "三三两两", "四平久稳", "五光十色"]
test_data_x = np.array(loadData(data_test))
test_data_x = test_data_x.reshape([-1, 12])
test_data_y = np.array(
    [template[1], template[0], template[0], template[1], template[1], template[1], template[0], template[1],
     template[0], template[1]])  # 1代表是成语，0代表不是
test_data_y = test_data_y.reshape([-1, 2])

# 测试隐藏层个数对结果的影响
for size in range(1, 20):
    y = run(size)
    # y = juanji(size)
    hidden_x.append(size)
    hidden_y.append(y)

n_hidden_size = hidden_x[hidden_y.index(max(hidden_y))]
print("use hidden size:%d to run" % (n_hidden_size))
exercise = run(n_hidden_size)
# exercise = juanji(n_hidden_size)
pre = run(8)
# pre = juanji(8)
print("exercise hidden size result is:%f, predict hidden size result is:%f" % (exercise, pre))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(hidden_x, hidden_y)
plt.show()
