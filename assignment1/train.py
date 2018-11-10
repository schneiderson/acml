import numpy as np
import NN
import matplotlib.pyplot as plt


def main():
    input_data = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

    nn = NN.NN()

    print("Input: \n" + str(input_data))
    print("Actual Output: \n" + str(input_data))

    xaxis = []
    errors = []
    predictions = []

    for i in range(20001):
        if i % 500 == 0:
            xaxis.append(i)
            print(" #" + str(i) + "\n")
            prediction = nn.predict(input_data).round(3)
            predictions.append(prediction)
            print("Predicted Output: \n" + str(prediction))
            error = np.mean(np.square(input_data - nn.predict(input_data))).round(4)
            errors.append(error)
            print("Loss: \n" + str(error))  # mean sum squared loss
            print("\n")

        nn.train(input_data, input_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(xaxis, np.array(errors))

    predictions = np.array(predictions)

    for i in range(8):
        ax2.plot(xaxis, predictions[:, i, i])

    print("\n\nweights layer 1:")
    print(nn.W1)
    print("\n\nbias weights layer 1:")
    print(nn.bw1)

    print("\n\nweights layer 2 (output):")
    print(nn.W2)
    print("\n\nbias weights layer 2 (output):")
    print(nn.bw2)

    print("\n\nhidden layer activations:")
    print(nn.layer1)

    print("\n\nhidden layer activations (rounded):")
    print(np.round(nn.layer1, 0))

    print("\n\noutput layer activations:")
    print(nn.output)

    plt.show()

    fig.savefig("graphs.pdf", bbox_inches='tight')


if __name__ == '__main__':
    main()
