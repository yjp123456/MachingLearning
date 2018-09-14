import os

import numpy as np
import tensorflow as tf
from RL_brain import DeepQNetwork
from maze_env import Maze

save_path = "my_net/save_net.ckpt"


# 旅游推荐模型训练

def run_maze():
    # 将上次保存的训练模型拿出来使用，可以放到客户端执行
    with open('./model/mnist.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        newInput_X = tf.placeholder(tf.float32, [None, RL.n_features], name='s')

        output = tf.import_graph_def(graph_def,
                                     input_map={'s:0': newInput_X},
                                     return_elements=['eval_net/Q/output:0'])

        for i in range(0, 10):
            actions_value = RL.sess.run(output, feed_dict={newInput_X: [[2, 2]]})
            action = np.argmax(actions_value)
            print('predict result:%s' % str(actions_value))

    # 导入上一次结果
    if os.listdir("./my_net"):
        print("train data exist, restore it")
        saver = tf.train.Saver()
        saver.restore(RL.sess, save_path)

    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action, observation)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game

    saver = tf.train.Saver()
    result = saver.save(RL.sess, save_path)  # 保存训练结果

    # 保存结果为pb文件，需要指定output_node_names
    output_graph_def = tf.graph_util.convert_variables_to_constants(RL.sess, RL.sess.graph_def,
                                                                    ['eval_net/Q/%s' % RL.output_node])
    with tf.gfile.FastGFile('model\mnist.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())
    print('game over, save result:%s' % result)
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)  # 定时器
    env.mainloop()
    RL.plot_cost()
