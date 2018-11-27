import tensorflow as tf
import numpy as np
import gym
#import matplotlib.pyplot as plt
from gym.wrappers import Monitor

EPISODES = 10000
episode_number = 0
env = gym.make('MountainCar-v0')
env.seed(1)
env = env.unwrapped

# create network graph
D = env.observation_space.shape[0]
A = env.action_space.n
H = 10
learning_rate = 0.02
gamma = 0.995

tf.reset_default_graph()
input_x = tf.placeholder(tf.float32, [None, D], name="input_x")
fc1 = tf.contrib.layers.fully_connected(inputs=input_x,
                                        num_outputs=H,
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())
fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                        num_outputs=A,
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())
fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                        num_outputs=A,
                                        activation_fn=None,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

output = tf.nn.softmax(fc3)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 3], name="input_y")
discounted_rewards = tf.placeholder(tf.float32, name="discounted_rewards")
neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=input_y)
product = neg_log_likelihood * discounted_rewards
loss = tf.reduce_mean(product)  # no need for -ve sign if using tf.nn.softmax_cross_entr.....
# as it gives neg_log_likelihood

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initialize_all_variables()

def discounted_rewards_(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r


def choose_action(output):
    action = np.random.choice(range(output.shape[1]), p=output.ravel())
    return action

rewards = []
episode_save = 100

with tf.Session() as sess:
    sess.run(init)

    xs, drs, ys = [], [], []

    reward_sum = 0
    current_state = env.reset()
    render = True
    done = False
    goal_reached = False

    while not goal_reached:
        if render and (episode_number + 1) % episode_save == 0:
            env.render()
        x = np.reshape(current_state, [1, D])
        out = sess.run(output, feed_dict={input_x: x})
        action = choose_action(out)
        xs.append(x)
        temp_y = np.zeros(A)
        temp_y[action] = 1
        ys.append(temp_y)
        next_state, reward, done, _ = env.step(action)
        drs.append(reward)
        reward_sum += reward

        # if episode ends, find discounted rewards and
        # find gradients for the episode
        if done:
            episode_number += 1
            epx = np.vstack(np.array(xs))
            epy = np.vstack(np.array(ys))
            epr = np.vstack(np.array(drs))

            discounted_rs = discounted_rewards_(drs)
            discounted_rs -= np.mean(discounted_rs)
            discounted_rs /= np.std(discounted_rs)

            xs, ys, drs = [], [], []

            sess.run([loss, optimizer], feed_dict={discounted_rewards: discounted_rs, input_x: epx, input_y: epy})
            print("Reward in episode :", episode_number, "is :", reward_sum)

            reward_sum = 0
            if (episode_number + 1) % 5 == 0:
                rewards.append(reward_sum)
                #fig, axs = plt.subplots(1, 1, figsize=(5.8, 5))
                resolution = 30
                Q = np.zeros((resolution, resolution))
                positions = np.linspace(-1.2, 0.6, num=resolution, endpoint=True)
                velocities = np.linspace(-0.7, 0.7, num=resolution, endpoint=True)
                for x in range(30):
                    for y in range(30):
                        curr_state = (positions[x], velocities[y])
                        x_vec = np.reshape(curr_state, [1, D])
                        out = sess.run(output, feed_dict={input_x: x_vec})

                        #

            current_state = env.reset()

        current_state = next_state


print(reward_sum)

env.close()












