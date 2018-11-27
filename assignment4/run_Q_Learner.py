
import gym
from assignment4.QLearner import QLearner as QL
import matplotlib.pyplot as plt

slow = False
print_to_cons = False
train = False
load_from_file = not train

directory = 'data4/'

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')

# env = gym.make("MountainCar-v0")

actions = [0, 1, 2]
gamma = 0.97
alpha = 0.4
epsilon = 1
epsilon_floor = 0.1
exploration_decay = 0.955
epochs = 50000

QL = QL(actions, epsilon, alpha, gamma, 27, 27)

# -------------------
# Training Code
# -------------------
if train:
    for e in range(epochs):
        observation = env.reset()
        done = False
        timesteps = 0

        while not done:
            if slow:
                env.render()
            action = QL.e_greedy(observation)
            new_observation, reward, done, info = env.step(action)
            timesteps += 1
            if slow:
                print(new_observation)
            if slow:
                print(reward)
            if slow:
                print(done)

            QL.learn_q(observation, action, reward, new_observation)
            observation = new_observation

        print("Episode finished after ", timesteps, "timesteps.")

        if reward > 0:
            if slow:
                print("#TRAIN Episode:{} finished after {} timesteps. Reached GOAL!.".format(e, timesteps))
        else:
            if slow:
                print("#TRAIN Episode:{} finished after {} timesteps.".format(e, timesteps))

        # Update exploration
        QL.epsilon *= exploration_decay
        QL.epsilon = max(epsilon_floor, QL.epsilon)
        #

        # Export data
        if e % 500 == 0:
            QL.export_q(directory + 'Q_table_%d_%d_3_epoch_%d' % (QL.n_positions, QL.n_velocities, e))

    print("Done!.")
    # Close environment
    env.close


# -----------------------
# Test Trained Q-Learner
# -----------------------
if load_from_file:
    epoch = 45000

    QL.import_q(directory + 'Q_table_27_27_3_epoch_' + str(epoch))
    QL.plot_q()

    for e in range(epochs):

        observation = env.reset()
        done = False
        timesteps = 0

        while not done:
            env.render()
            action = QL.greedy(observation)
            new_observation, reward, done, info = env.step(action)
            timesteps += 1
            print(new_observation)
            print(reward)
            print(done)

            observation = new_observation

        print("Episode finished after ", timesteps, "timesteps.")

        if reward > 0:
            print("#TRAIN Episode:{} finished after {} timesteps. Reached GOAL!.".format(e, timesteps))
        else:
            print("#TRAIN Episode:{} finished after {} timesteps.".format(e, timesteps))


    print("Done!.")
    # Close environment
    env.close
