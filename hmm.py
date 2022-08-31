import numpy as np


def forward(y, A, B, π):
    """
    Forward algorithm, compute likelihood of given observed vector
    :param y: Observation vector
    :param A: State transition matrix
    :param B: Emission matrix
    :param π: Initial probability matrix
    """
    n_states = A.shape[0]
    T = y.size  # sequence length
    α = np.zeros(shape=(T, n_states))  # forward probability

    # base cases
    for i in range(n_states):
        α[0][i] = π[i] * B[i, y[0]]

    # recursion
    for t in range(1, T):
        α[t, :] = sum(α[t - 1][j] * A[j, :] * B[:, y[t]] for j in range(n_states))

    likelihood = sum(α[-1, :])
    return likelihood


def viterbi(y, A, B, π):
    """
    Viterbi algorithm, find MAP hidden state vector path
    :param y: Observation vector
    :param A: State transition matrix
    :param B: Emission matrix
    :param π: Initial probability matrix
    """
    n_states = A.shape[0]
    T = y.size  # sequence length
    α = np.zeros(shape=(T, n_states))  # forward probability
    β = np.zeros(shape=(T, n_states), dtype=int)  # back pointers

    # base cases
    for i in range(n_states):
        α[0][i] = π[i] * B[i, y[0]]

    # recursion
    for t in range(1, T):
        branches = np.array([α[t - 1][j] * A[j, :] * B[:, y[t]] for j in range(n_states)])
        α[t] = np.max(branches, axis=0)
        β[t] = np.argmax(branches, axis=0)

    # build best path
    best_path = []
    last_pointer = np.argmax(α[-1, :])
    for t in reversed(range(0, T)):
        best_path.append(last_pointer)
        last_pointer = β[t][last_pointer]
    best_path = np.array(np.flip(best_path))

    return best_path, α, β
