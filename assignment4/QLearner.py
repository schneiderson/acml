import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fig, axs = plt.subplots(1, 1, figsize=(5.8, 5))


class QLearner(object):
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9, n_positions=27, n_velocities=27):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

        # Q-table
        self.p_lim = [-1.2, 0.6]
        self.v_lim = [-0.07, 0.07]
        self.n_positions = n_positions
        self.n_velocities = n_velocities
        # discretize the positions
        self.positions = np.linspace(self.p_lim[0], self.p_lim[1], num=self.n_positions, endpoint=True)
        self.velocities = np.linspace(self.v_lim[0], self.v_lim[1], num=self.n_velocities, endpoint=True)

        self.Q = (-1) * np.ones((self.n_velocities, self.n_positions * len(self.actions)))

        self.plot_reset = True

    def get_q(self, state, action):
        # get position our position is closest too in descretization
        pos = np.argmin(abs(self.positions - state[0]), axis=0)
        # get velocity our velocity is closest too in descretization
        vel = np.argmin(abs(self.velocities - state[1]), axis=0)

        return self.Q[vel, 3 * pos + action]

    def learn_q(self, state1, action1, reward, state2):
        # Q-learning: Q(s,a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        pos = np.argmin(abs(self.positions - state1[0]), axis=0)
        vel = np.argmin(abs(self.velocities - state1[1]), axis=0)

        Q_n = self.Q[vel, 3 * pos + action1]

        maxQ = max([self.get_q(state2, a) for a in self.actions])

        self.Q[vel, 3 * pos + action1] = Q_n + self.alpha * (reward + self.gamma * maxQ - Q_n)

    def e_greedy(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.greedy(state)
        return action

    def greedy(self, state):
        q = [self.get_q(state, a) for a in self.actions]
        maxQ = max(q)

        best = [i for i in range(len(self.actions)) if q[i] == maxQ]

        # if there are several state-action maximal values, select one of them randomly
        if len(best) > 1:
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        return self.actions[i]

    def export_q(self, fname):
        print('--------------------------------------------------')
        print(self.Q)
        print('--------------------------------------------------')

        print("Export Q-table to {}".format(fname))
        np.save(fname, self.Q)

    def import_q(self, fname):
        print("Import Q-table from {}".format(fname))
        self.Q = np.load(fname + '.npy')
        print('--------------------------------------------------')
        print(self.Q)
        print('--------------------------------------------------')

    def print_q(self):
        print('--------------------------------------------------')
        print(self.Q)

    def plot_q(self, clear=False):
        # Parameters
        grid_on = True
        v_max = 10.  # np.max(self.Q[0, :, :])
        v_min = -50.
        x_labels = ["%.2f" % x for x in self.positions]
        y_labels = ["%.2f" % y for y in self.velocities]
        titles = "Actions " + u"\u25C0" + ":push_left/" + u"\u25AA" + ":no_push/" + u"\u25B6" + ":push_right"
        Q = np.zeros((self.n_velocities * len(self.actions), self.n_positions * len(self.actions)))

        for s_2 in range(len(self.velocities)):
            Q[3 * s_2 + 0, :] = self.Q[s_2, :]
            Q[3 * s_2 + 1, :] = self.Q[s_2, :]
            Q[3 * s_2 + 2, :] = self.Q[s_2, :]

        im = axs.imshow(Q, interpolation='nearest', vmax=v_max, vmin=v_min, cmap=cm.jet)
        axs.grid(grid_on)
        axs.set_title(titles)
        axs.set_xlabel('Position')
        axs.set_ylabel('Velocity')
        x_start, x_end = axs.get_xlim()
        #y_start, y_end = axs.get_ylim()
        axs.set_xticks(np.arange(x_start, x_end, 3))
        axs.set_yticks(np.arange(x_start, x_end, 3))
        axs.set_xticklabels(x_labels, minor=False, fontsize='small', horizontalalignment='left', rotation=90)
        axs.set_yticklabels(y_labels, minor=False, fontsize='small', verticalalignment='top')
        self.cb = fig.colorbar(im, ax=axs)
        #
        plt.show(block=False)

    def save_plot(self, fname):
        fig.savefig(fname, bbox_inches='tight')