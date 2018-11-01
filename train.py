import numpy as np
import NN


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

    for i in range(1000):
        print(" #" + str(i) + "\n")
        print("Predicted Output: \n" + str(nn.predict(input_data)))
        print("Loss: \n" + str(np.mean(np.square(input_data - nn.predict(input_data)))))  # mean sum squared loss
        print("\n")

        nn.train(input_data, input_data)


if __name__ == '__main__':
    main()
