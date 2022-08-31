import numpy as np
from hmm import forward, viterbi

if __name__ == '__main__':
    A = np.array([[0.5, 0.5], [0.3, 0.7]])  # transition matrix
    B = np.array([[0.8, 0.2], [0.4, 0.6]])  # emission matrix
    π = np.array([0.375, 0.625])  # initial probability distribution
    T = 10  # sequence length
    y = np.random.randint(0, 2, T)

    l = forward(y, A, B, π)
    best_path, α, β = viterbi(y, A, B, π)
    print(l)
    print(best_path)
