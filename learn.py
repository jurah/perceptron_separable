debug = False  # use True for debug prints

with open('data/linsep-traindata.csv') as f:
    examples = [[float(x) for x in example.rstrip('\n').split(',')] for example in f.readlines()]

with open('data/linsep-trainclass.csv') as f:
    labels = [float(label.rstrip('\n')) for label in f.readlines()]


def predict(w, e):
    if w[0] * e[0] + w[1] * e[1] >= 0.0:
        return 1.0
    else:
        return -1.0


num_examples = len(examples)

weights = [1.0, 10.0]  # [0, 0] too good initialization
num_last_correctly_classified = 0
i = 0
learning_rate = 0.1


# stochastic gradient descent, iterating cyclically through training examples
# assuming they are separable, we can require to correctly classify all training examples
while (num_last_correctly_classified < num_examples):
    example = examples[i]
    label = labels[i]
    if debug:
        print "example, label, prediction: ", example, label, predict(weights, example)
    if predict(weights, example) != label:
        num_last_correctly_classified = 0
        weights[0] -= learning_rate * example[0]
        weights[1] -= learning_rate * example[1]
        if debug:
            print "weights after update: ", weights

    num_last_correctly_classified += 1
    i = (i + 1) % num_examples

print "-------------------------"
print "final weights: ", weights
