import numpy as np

class Generator:
    def __init__(self, num_worker=100, num_item=1000, num_category=2, alpha=2, beta=2):
        self.num_worker = num_worker
        self.num_item = num_item
        self.num_category = num_category
        self.alpha = alpha
        self.beta = beta

    def confusionMatrix(self, true_prob=np.array([0.3, 0.9]),
                        false_prob=np.array([0.0, 0.5]),
                        all_same_prop=0.02,
                        all_false_prop=0.02):
        num_all_same = int(self.num_worker * all_same_prop)
        num_all_false = int(self.num_worker * all_false_prop)
        num_normal = self.num_worker - num_all_same - num_all_false
        def normal_worker(num, num_category):
            CM = np.random.rand(num, num_category, num_category)
            CM = CM * (false_prob[1]-false_prob[0]) + false_prob[0]
            [np.fill_diagonal(m, np.random.rand(self.num_category) * (true_prob[1]-true_prob[0]) + true_prob[0]) for m in CM]
            CM = np.array([m / np.sum(m, axis=1).reshape(-1, 1) for m in CM])
            return CM

        def all_same_worker(num, num_category):
            if num < 2:
                return np.array([]).reshape(-1, num_category, num_category)
            CM = np.zeros((num, num_category, num_category))
            random_label = np.random.randint(0, num_category, num)
            for i, k in enumerate(random_label):
                CM[i, :, k] = 1
            return CM

        def all_false_worker(num, num_category):
            if num < 2:
                return np.array([]).reshape(-1, num_category, num_category)
            CM = np.random.rand(num, num_category, num_category)
            CM = CM * (false_prob[1] - false_prob[0]) + false_prob[0]
            [np.fill_diagonal(m, 0) for m in CM]
            CM = np.array([m / np.sum(m, axis=1).reshape(-1, 1) for m in CM])
            return CM

        worker1 = normal_worker(num_normal, self.num_category)
        worker2 = all_same_worker(num_all_same, self.num_category)
        worker3 = all_false_worker(num_all_false, self.num_category)

        return np.concatenate((worker1, worker2, worker3))

    def generate_item_label(self, prob=None):
        if prob is None:
            prob = np.array([0.5, 0.5])
        elif isinstance(prob, float):
            prob = np.array([prob, 1-prob])
        if len(prob) != self.num_category:
            raise ValueError('length of probability list should equal to number of category')
        label = np.random.choice(np.arange(1, self.num_category+1), size=self.num_item, replace=True, p=prob)
        item = np.arange(1, self.num_item+1)
        self.truth = np.stack((item, label), axis=1)
        return self.truth

    def generate_worker_label(self, CM, truth):
        worker_p = np.random.beta(self.alpha, self.beta, self.num_worker)
        res = []
        for i in range(1, self.num_item+1):
            worker_list = np.arange(1, self.num_worker+1)[np.random.rand(self.num_worker) < worker_p]
            real_label = truth[i-1, 1]
            for j in worker_list:
                label = np.random.choice(np.arange(1, self.num_category+1), size=1, p=CM[j-1, real_label-1, :])[0]
                res.append([i, j, label])
        self.label = np.array(res)
        return self.label

if __name__ == '__main__':
    g = Generator(num_worker=10, num_item=10)
    CM = g.confusionMatrix()
    # print(g.confusionMatrix())
    truth = g.generate_item_label()
    # print(truth)

    label = g.generate_worker_label(CM, truth)
    # print(label)
    np.savetxt('synthetic_data/1_truth.txt', truth, delimiter=' ', fmt='%d')
    np.savetxt('synthetic_data/1_crowd.txt', label, delimiter=' ', fmt='%d')










