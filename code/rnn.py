import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001  # 学习率
training_iters = 100000  # 总数据量大小
batch_size = 128  # 每次训练数据量大小

n_inputs = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # 步长
n_hidden_units = 128  # neurons in hidden layer
n_classes = 10  # 最终输出结果分为0-9十种

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  # 输入大小，即n_steps x n_inputs
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# LSTM循环神经网络
def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################
    # RNN分为三个门
    # 1、输入门控制，第一，sigmoid 层称 “输入门层” 决定什么值我们将要更新。然后，一个 tanh 层创建一个新的候选值向量
    # 2、忘记门控制，它会读取前面累积的结果和当前输入，然后输出一个在 0 到 1 之间的数值来决定之前结果的保留比例。1 表示“完全保留”，0 表示“完全舍弃”
    # 3、输出门控制，整合输入门和忘记门的结果

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)，二维数组
    X = tf.reshape(X, [-1, n_inputs])  # 将原数据重组成固定n_inputs列的数组，-1代表行数自动计算

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)，二维数组
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)，三维数组
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.忘记门控制，它会读取前面累积的结果和当前输入，然后输出一个在 0 到 1 之间的数值来决定之前结果的保留比例。1 表示“完全保留”，0 表示“完全舍弃”
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)  # 初始化为0

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # 如果 inputs 为 (batches, steps, inputs) ==> time_major=False;如果 inputs 为 (steps, batches, inputs) ==> time_major=True;
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)  # 动态rnn不需要预编译，速度更快

    # 将结果显示到histogram图表里
    tf.summary.histogram('/output_all', outputs)
    tf.summary.histogram('/output', outputs[-1])

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        # x = [[[1,2,3,4],[5,6,7,8],[5,6,7,8]],[[9,12,13,14],[15,16,17,18],[5,6,7,8]]]
        # 此3维数组为2x3x4，可以看成是两个 3x4的二维数组
        # [0,1,2],0代表三维数组的高（即为二维数组的个数），1代表二维数组的行，2代表二维数组的列，
        # tf.transpose(x, perm=[1,0,2])代表将三位数组的高和行进行转置
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    # shape = (128, 10)，tf.matmul是矩阵相乘，output是一个三维数组，outputs[-1]是取output最后一个元素，每个元素是一个二维数组
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    train_writer = tf.summary.FileWriter("logs/output", sess.graph)
    merged = tf.summary.merge_all()
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })

        if step == 20:
            train_result = sess.run(merged, feed_dict={
                x: batch_xs,
                y: batch_ys,
            })
            train_writer.add_summary(train_result, step)
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
        step += 1
