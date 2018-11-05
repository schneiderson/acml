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

    #print("Input: \n" + str(input_data))
    #print("Actual Output: \n" + str(input_data))
    iterations = 200000
    numOfTests = 8
    for i in range(iterations):
        # big_delta_input = np.zeros((9, 3))
        # big_delta_hidden = np.zeros((4, 8))
        # for testNumber in range(numOfTests):
        if i == iterations-1:
            print(" #" + str(i) + "\n")
            print("Predicted Output: \n" + str(nn.predict(input_data).round(3)))
            print("Loss: \n" + str(np.mean(np.square(input_data - nn.predict(input_data)))))  # mean sum squared loss
            print("\n")
            print("\n")
            print("Hidden layer\n" + str(nn.layer1.round(3)))

        nn.train(input_data, input_data, numOfTests)

if __name__ == '__main__':
    main()
