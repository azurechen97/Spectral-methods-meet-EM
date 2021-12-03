import numpy as np

class spectralEM:
    def __init__(self, init_mu, labels):
        """
        class initialization
        :param init_mu: initialization mu, mu is confusion matrix with dimension M workers, K labels, K labels
        :param labels: crowed sourcing labels result, dimension M workers, N items
        """
        self.mu = init_mu
        self.M, self.N = labels.shape
        self.K = np.max(labels)+1
        # self.N = num_item
        # self.M = num_worker
        # self.K = num_category
        self.q = np.zeros((self.N, self.K)) # / self.K
        self.labels = labels

    def E_step(self):
        log_mu = np.log(np.clip(self.mu, 1e-6, 10))
        I, J = self.labels.shape
        for j in range(J):
            for l in range(self.K):
                for i in range(I):
                    label = self.labels[i, j]
                    if label != -1:
                        self.q[j, l] += log_mu[i, l, label]
            self.q[j, :] = np.exp(self.q[j, :])
            self.q[j, :] = self.q[j, :] / np.sum(self.q[j, :])

    def M_step(self):
        for i in range(self.M):
            for l in range(self.K):
                for c in range(self.K):
                    self.mu[i, l, c] = np.sum(self.q[:, l] * (self.labels[i, :] == c))
                self.mu[i, l, :] = self.mu[i, l, :] / np.sum(self.mu[i, l, :])

    def run(self, strategy='max_iter', max_iter=10, delta=1e-6):
        self.E_step()
        d = 1
        mu0 = self.mu.copy()
        num_iter = 0
        while (strategy=='max_iter' and num_iter < max_iter) or (strategy=='converge' and d > delta):
            self.M_step()
            self.E_step()
            num_iter += 1
            if strategy == 'converge' :
                mu = self.mu.copy()
                d = np.sum(np.abs(mu - mu0)) / self.N
                mu0 = mu

        print('# iterations = ', num_iter)


    def output_mu(self):
        return self.mu

    def output_q(self):
        return self.q







