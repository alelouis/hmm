import numpy as np


def forward(y, A, B, π):
    """
    Forward algorithm, compute likelihood of given observed vector
    https://web.stanford.edu/~jurafsky/slp3/A.pdf
    :param y: Observation vector
    :param A: State transition matrix
    :param B: Emission matrix
    :param π: Initial probability matrix
    """
    n_states = A.shape[0]
    T = y.size  # sequence length
    α = np.zeros(shape=(T, n_states))  # forward probability

    # base cases
    α[0] = π * B[:, y[0]]

    # recursion
    for t in range(1, T):
        α[t, :] = sum(α[t - 1][j] * A[j, :] * B[:, y[t]] for j in range(n_states))

    likelihood = sum(α[-1, :])
    return likelihood


def viterbi(y, A, B, π):
    """
    Viterbi algorithm, find MAP hidden state vector path
    https://web.stanford.edu/~jurafsky/slp3/A.pdf
    :param y: Observation vector
    :param A: State transition matrix
    :param B: Emission matrix
    :param π: Initial probability matrix
    """
    n_states = A.shape[0]
    T = y.size  # sequence length
    v = np.zeros(shape=(T, n_states))  # forward probability
    b = np.zeros(shape=(T, n_states), dtype=int)  # back pointers

    # base cases
    v[0] = π * B[:, y[0]]

    # recursion
    for t in range(1, T):
        branches = np.array([v[t - 1][j] * A[j, :] * B[:, y[t]] for j in range(n_states)])
        v[t] = np.max(branches, axis=0)
        b[t] = np.argmax(branches, axis=0)

    # build best path
    best_path = []
    last_pointer = np.argmax(v[-1, :])
    for t in reversed(range(0, T)):
        best_path.append(last_pointer)
        last_pointer = b[t][last_pointer]
    best_path = np.array(np.flip(best_path))

    return best_path, v, b
