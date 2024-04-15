from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]



print("\n<<<<<<< 2 HIDDEN NODES >>>>>>>\n")

xor_neural_net = NeuralNet(2, 2, 1)
xor_neural_net.train(xor_training_data, iters=50000)

# After 9300 iterations, the change in error converged by reaching a value less than 0.0005
# The actual value after after those iterations was 0.0004976351530286594
# I ran it with 50000 iterations and reached a change in error of 8.242983108703615e-05

print(xor_neural_net.get_ih_weights())
print()
print(xor_neural_net.get_ho_weights())



print("\n<<<<<<< 8 HIDDEN NODES >>>>>>>\n")

xor_neural_net = NeuralNet(2, 8, 1)
xor_neural_net.train(xor_training_data, iters=50000)

# After 7700 iterations, the change in error converged by reaching a value less than 0.0005
# The actual value after after those iterations was 0.0004958616290734465
# I ran it with 50000 iterations and reached a change in error of 6.354714023156375e-05

print(xor_neural_net.get_ih_weights())
print()
print(xor_neural_net.get_ho_weights())



print("\n<<<<<<< 1 HIDDEN NODE >>>>>>>\n")

xor_neural_net = NeuralNet(2, 1, 1)
xor_neural_net.train(xor_training_data, iters=50000)

# After 50000 iterations, the change in error never converged and stayed at around 0.34
# This is because the XOR function cannot be computed with only one hidden layer

print(xor_neural_net.get_ih_weights())
print()
print(xor_neural_net.get_ho_weights())

