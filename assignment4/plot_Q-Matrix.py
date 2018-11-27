from assignment4.QLearner import QLearner as QL

actions = [0, 1, 2]
gamma = 0.97
alpha = 0.2
epsilon = 1
epsilon_floor = 0.1
exploration_decay = 0.955
epochs = 1000
directory = 'data2/'

QL = QL(actions, epsilon, alpha, gamma, 27, 27)

epoch = 5000
QL.import_q(directory + 'Q_table_27_27_3_epoch_' + str(epoch))

QL.plot_q()
QL.save_plot(directory + "Q_table_27_27_3_epoch_" + str(epoch) + ".pdf")