import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

# k: number of classes
# labels: a mxn matrix, labels of n items by m workers
# Zg: a 3xnxk tensor, the group aggregated labels in frequency
# groups: a m vector, each worker in which group
def get_Zg(k, labels, groups):
    m, n = labels.shape
    Zg = np.zeros((3, n, k))
    for g in range(3):
        for l in range(k):
            Zg[g, :, l] = np.mean(labels[groups == g, :] == l, axis=0)
    return Zg

# Z: a 3xnxk tensor, the group aggregated labels in frequency
# M2s: a list of the second moments (kxk)
# M3s: a list of the third moments (kxkxk)
def get_M(Zg):
    _, n, k = Zg.shape
    M2s, M3s = [], []
    for a, b, c in [(1, 2, 0), (2, 0, 1), (0, 1, 2)]:
        Za = (Zg[c, :, :].T@Zg[b, :, :]/n)@np.linalg.inv(Zg[a, :, :].T @
                                                       Zg[b, :, :]/n)@Zg[a, :, :].T
        Zb = (Zg[c, :, :].T@Zg[a, :, :]/n)@np.linalg.inv(Zg[b, :, :].T @
                                                       Zg[a, :, :]/n)@Zg[b, :, :].T
        M2s.append(Za@Zb.T/n)
        M3s.append(np.einsum("ji,ki,li->jkl", Za, Zb, Zg[c, :, :].T)/n)
    return M2s, M3s

# get whitten matrix Q
# sym: whether to symmetricalize M2
def get_whiten(M2, sym=True):
    # closest symmetric matrix
    if sym:
        M2 = (M2+M2.T)/2
    u, s, vh = np.linalg.svd(M2, full_matrices=False)
    return u@np.diag(s**-0.5)

# whiten a tensor M3 using Q
def whiten_tensor(M3, Q):
    return np.einsum("def,da,eb,fc->abc", M3, Q, Q, Q)

# symmetricalize 3 dimensional tensor T
def sym_tensor(T):
    return (T+T.transpose(0, 2, 1)+T.transpose(1, 0, 2)+T.transpose(1, 2, 0)+T.transpose(2, 0, 1)+T.transpose(2, 1, 0))/6

# Robust tensor power method to calculate the eigensystem of a tensor
# L: number of candidates
# N: number of iterations
def robust_tensor_power(T, L=20, N=100, sym=True):
    k, _, _ = T.shape
    if sym:
        T = sym_tensor(T)
    values = np.zeros(k)
    vectors = np.zeros((k, k))
    for h in range(k):
        theta_tau = []
        for tau in range(L):
            theta = np.random.randn(k)
            theta /= np.linalg.norm(theta)
            for t in range(N):
                theta = np.einsum("aef,e,f->a", T, theta, theta)
                theta /= np.linalg.norm(theta)
            theta_tau.append(theta)

        lam_best = float("-Inf")
        theta_best = np.array([])
        for theta in theta_tau:
            lam = np.einsum("efg,e,f,g->", T, theta, theta, theta)
            if lam > lam_best:
                lam_best = lam
                theta_best = theta

        theta = theta_best
        for t in range(N):
            theta = np.einsum("aef,e,f->a", T, theta, theta)
            theta /= np.linalg.norm(theta)
        lam = np.einsum("efg,e,f,g->", T, theta, theta, theta)

        values[h] = lam
        vectors[:, h] = theta
        T = T-lam*np.einsum("a,b,c->abc", theta, theta, theta)
    return values, vectors

# get the estimated confusion matrix
# note that each column corresponds a true label, which is different from scikit-learn
def get_confusion_matrix(k, labels, groups=None, sym=True, method = 0, cutoff=1e-7, L=50, N=10, seed=None):
    m, n = labels.shape
    if seed is not None:
        np.random.seed(seed)
    if groups is None:
        groups = np.random.randint(3, size=m)
    Zg = get_Zg(k, labels, groups)
    M2s, M3s = get_M(Zg)
    Cc = np.zeros((3, k, k))
    W = np.zeros((3, k))
    for g, (M2, M3) in enumerate(zip(M2s, M3s)):
        Q = get_whiten(M2, sym)
        M3_whiten = whiten_tensor(M3, Q)
        values, vectors = robust_tensor_power(M3_whiten, L, N, sym)
        w = values**-2
        mu = np.linalg.inv(Q.T)@vectors@np.diag(values)
        best = np.argmax(mu, axis=0)

        # the method in original code
        if method == 0:
            for h in range(k):
                l = best[h]
                if W[g, l] != 0:
                    l = np.where(W[g, :] == 0)[0][0]
                Cc[g, :, l] = mu[:, h].ravel()
                W[g, l] = w[h]
        # prevent multiple mu in same column
        else:
            not_in_best = []
            not_used_loc = np.array([], dtype=np.int64)
            for l in range(k):
                loc = np.where(best == l)[0]
                if len(loc) == 1:
                    Cc[g, :, l] = mu[:, loc].ravel()
                    W[g, l] = w[loc]
                elif len(loc) == 0:
                    not_in_best.append(l)
                else:
                    chosen = np.random.randint(len(loc))
                    Cc[g, :, l] = mu[:, loc[chosen]].ravel()
                    W[g, l] = w[loc[chosen]]
                    not_used_loc = np.append(not_used_loc, np.delete(loc, chosen))
            not_used_loc = np.random.permutation(not_used_loc)
            for i, l in enumerate(not_in_best):
                Cc[g, :, l] = mu[:, not_used_loc[i]].ravel()
                W[g, l] = w[not_used_loc[i]]

    W = np.mean(W, axis=0)
    C = np.zeros((m, k, k))
    for i in range(m):
        Ca = (np.sum(Cc, axis=0)-Cc[groups[i], :, :])/2
        Za = (np.sum(Zg, axis=0)-Zg[groups[i], :, :])/2
        E = np.zeros((k, k))
        for j in range(n):
            if labels[i, j] != -1:
                E[labels[i, j], :] += Za[j, :]
        E /= n
        Ci = E@np.linalg.inv(W[np.newaxis, :]*Ca.T)
        if cutoff:
            Ci[Ci < cutoff] = cutoff
        colsums = np.sum(Ci, axis=0)
        Ci /= colsums[np.newaxis, :]
        C[i, :, :] = Ci
    return C
    
# transform the data table into a mxn label matrix
def transform_data(data):
    X = np.array(data, dtype=np.int64)
    rows, _ = X.shape
    n, m, k = map(int, np.max(np.array(data), axis=0))
    # if the worker didn't label an item, then the label is written as -1
    labels = np.zeros((m, n), dtype=np.int64)-1
    for r in range(rows):
        labels[X[r, 1]-1, X[r, 0]-1] = X[r, 2]-1
    return labels

def errorRate(pred_q, truth):
    pred_label = np.argmax(pred_q, axis=1)
    y = truth[:,1] - 1
    return np.mean(pred_label != y)

# get the true confusion matrix
def get_true_confusion_matrix(data, truth, normalize=True):
    truth = truth.drop_duplicates()
    merged = data.merge(truth, on=0)
    n, m, k = np.max(np.array(data), axis=0)
    C = np.zeros((m, k, k))
    for i in range(m):
        merged_i = merged[merged["1_x"] == i+1]
        Ci = confusion_matrix(merged_i["1_y"], merged_i[2], labels=[
                              i+1 for i in range(k)]).T
        if normalize:
            Ci = Ci.astype(float)
            colsums = np.sum(Ci, axis=0)
            Ci[:, colsums != 0] /= colsums[np.newaxis, colsums != 0]
            Ci[:, colsums == 0] = 1./k
        C[i, :, :] = Ci
    return C

# the mse of estimated confusion matrix
def confusion_matrix_loss(C_estimated, C_true):
    return np.mean((C_estimated-C_true)**2)
