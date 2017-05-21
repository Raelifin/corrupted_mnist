# corrupted_mnist
MNIST Classifier on Corrupted Digits.

`data.csv` is a copy of MNIST with 10% of the data corrupted such that the entry
is a blend of two digits. This code uses tensorflow to train a model which then
only guesses when it's confident that it knows the answer.
