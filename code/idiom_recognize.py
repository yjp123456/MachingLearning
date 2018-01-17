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
        ys = tf.placeholder(tf.float32, [None, 1])

    # 定义隐藏层，这里使用Tensorflow 自带的激励函数tf.nn.relu
    L1 = add_layer(xs, 12, size, n_layer=1, activation_function=tf.nn.relu, norm=True)  # 定义输出层
    prediction = add_layer(L1, size, 1, n_layer=2, activation_function=None, norm=True)

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
            loss_value = sess.run(loss, feed_dict={xs: test_data_x, ys: test_data_y})
            print("预测值：%s" % (prediction_value))
            print("损失值：%s" % (loss_value))
            total += loss_value
            # plot the prediction
    return total / 20



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
                result_y.append(int(str[1]))
    finally:
        file_object.close()
    print(result_x)
    print(result_y)
    return result_x, result_y


# 识别输入的四个汉字是不是成语，仅供调研，实际并不能准确判断是不是成语
x_data, y_data = readIdiom()
x_data = np.array(loadData(x_data))
x_data = x_data.reshape([-1, 12])
y_data = np.array(y_data)
y_data = y_data.reshape([-1, 1])
hidden_x = []
hidden_y = []
n_hidden_size = 6

data_test = ["千方百计", "测试测试", "一五一十", "千篇一律", "万无一失",  "三三两两",  "五光十色"]
test_data_x = np.array(loadData(data_test))
test_data_x = test_data_x.reshape([-1, 12])
test_data_y = np.array([1, 0, 1, 1, 1, 1, 1])  # 1代表是成语，0代表不是
test_data_y = test_data_y.reshape([-1,1])

# 测试隐藏层个数对结果的影响
for size in range(1, 20):
    y = run(size)
    hidden_x.append(size)
    hidden_y.append(y)

n_hidden_size = hidden_x[hidden_y.index(min(hidden_y))]
print("use hidden size:%d to run" % (n_hidden_size))
exercise = run(n_hidden_size)
pre = run(8)
print("exercise hidden size result is:%f, predict hidden size result is:%f" % (exercise, pre))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(hidden_x, hidden_y)
plt.show()
