import numpy as np

class spectralEM:
    def __init__(self, init_mu, labels, num_item, num_worker, num_category, max_iter=10):
        self.mu = init_mu
        self.N = num_item
        self.M = num_worker
        self.K = num_category
        self.max_iter = max_iter
        self.q = np.ones((self.N, self.K)) / self.K
        self.labels = labels

    def E_step(self):
        log_mu = np.log(self.mu)
        I, J = self.labels.shape
        for j in range(J):
            for i in range(I):
                label = self.labels[i, j]
                for l in range(self.K):
                    if label != -1:
                        self.q[j, l] = np.exp(np.sum(log_mu[:, l, label]))
            self.q[j, :] = self.q[j, :] / np.sum(self.q[j, :])

    def M_step(self):
        for i in range(self.M):
            for l in range(self.K):
                for c in range(self.K):
                    self.mu[i, l, c] = np.sum(self.q[:, l] * (self.labels[i, :] == c))
                self.mu[i, l, c] = self.mu[i, l, c] / np.sum(self.mu[i, l, :])

    def run(self):
        while self.max_iter > 0:
            self.E_step()
            self.M_step()
            self.max_iter -= 1

    def output_mu(self):
        return self.mu

    def output_res(self):
        pass





