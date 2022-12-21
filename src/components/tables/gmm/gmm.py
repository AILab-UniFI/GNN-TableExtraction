import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
import scipy.cluster.vq as vq
import numpy.linalg as la
import numpy as np

class GMM(object):

    def __init__(self, k, data, init_method='random', min_sigma=1e-3, mu=None, sigma=None):

        """
        :param k:  number of components
        :param data: data, 1d numpy array
        :param init_method:
        :param mu:  np.arrays, means of gaussians components
        :param sigma: np.arrays, stds of gaussians components
        """
        # check parameters
        assert init_method in ['random','uniform','kmeans']
        if mu is not None: assert len(mu) == k
        if sigma is not None: assert len(sigma) == k
        assert k == int(k) and k >= 1

        self.K = k
        self.mu = [] if mu is None else mu
        self.sigma = [] if sigma is None else sigma
        self.data = np.array(data, dtype=np.float)
        assert len(self.data.shape) == 1

        self.gamma_mat = np.zeros((self.K, len(self.data))) # p(y = k|x = i, mu, sigma)
        self.min_sigma = min_sigma

        if mu is not None and sigma is not None:
            pass

        elif init_method is "uniform":
            # uniformly assign data points to components then estimate the parameters
            np.random.shuffle(self.data)
            n = len(self.data)
            s = n // self.K
            for i in range(self.K):
                self.mu.append(np.mean(self.data[i * s: (i + 1) * s], axis=0))
                self.sigma.append(np.std(self.data[i * s: (i + 1) * s]))

            self.priors = np.ones(self.K, dtype=np.float) / self.K

        elif init_method is "random":
            # choose ncomp points from data randomly then estimate the parameters
            mus = np.random.choice(data, self.K, replace=False)
            clusters = [[] for i in range(self.K)]
            for d in data:
                i = np.argmin([la.norm(d - m) for m in mus])
                clusters[i].append(d)

            print('clusters:\n',clusters)
            self.mu = [np.mean(clusters[i], axis=0) for i in range(self.K)]
            self.sigma = [np.std(clusters[i]) if len(clusters[i]) > 1 else self.min_sigma for i in range(self.K)]

            # self.priors = np.ones(self.K, dtype=np.float) / np.array([len(c) for c in clusters])
            self.priors = np.array([len(c) for c in clusters])


        elif init_method is "kmeans":
            # use kmeans to initialize the parameters
            (centroids, labels) = vq.kmeans2(self.data, self.K, minit="points", iter=100)
            clusters = [[] for i in range(self.K)]
            for (l, d) in zip(labels, self.data):
                clusters[l].append(d)

            print('clusters:\n',clusters)

            self.mu = [np.mean(clusters[i], axis=0) for i in range(self.K)]
            self.sigma = [np.std(clusters[i]) if len(clusters[i]) > 1 else self.min_sigma for i in range(self.K)]
            self.priors = np.array([len(c) for c in clusters])
            # self.priors = np.ones(self.K, dtype=np.float) / self.K


        self.mu = np.array(self.mu, dtype=float)
        self.sigma= np.array(self.sigma, dtype=float)
        self.priors = self.priors / np.sum(self.priors) # normalize priors

        print('mu: ', self.mu)
        print('sigma: ', self.sigma)
        print('prior: ', self.priors)


    def E(self):

        for i in range(len(self.data)):
            x_i = self.data[i]
            for k in range(self.K):
                p = self.single_variable_gausian(x_i, self.mu[k], self.sigma[k])
                self.gamma_mat[k, i] = self.priors[k] * p

        self.gamma_mat /= np.sum(self.gamma_mat, axis=0) # normalize

    def M(self):

        N = np.sum(self.gamma_mat, axis=1)

        for k in range(self.K):
            mu = np.dot(self.gamma_mat[k], self.data) / N[k]
            sigma = np.zeros_like(self.sigma[0])

            for i in range(len(self.data)):
                sigma += self.gamma_mat[k, i] * np.outer(self.data[i] - mu, self.data[i] - mu)

            sigma = sigma / N[k]
            self.sigma[k] = sigma
            self.mu[k] = mu
            self.priors[k] = N[k] / np.sum(N)  # normalize the new priors

    def log_likelyhood(self):
        return np.sum(np.log(self.priors.dot(self.gamma_mat)))

    def train(self, iter):

        for i in range(iter):
            self.E()
            self.M()
            print('current log likelyhood: {}, \ncurrent means \n{}, \nsigmas: \n{}'.format(self.log_likelyhood(), self.mu, self.sigma))

    def draw(self):
        import matplotlib.pyplot as plt
        x = np.linspace(min(self.data), max(self.data), 1000, endpoint=True)
        sum_yp = np.zeros_like(self.data)
        sum_y = np.zeros_like(x)
        plt.figure(0)
        plt.title('componets')
        for k in range(self.K):
            y = self.priors[k] * multivariate_normal.pdf(x, self.mu[k], self.sigma[k], allow_singular=True)
            sum_y += y
            yp = self.priors[k] * multivariate_normal.pdf(self.data, self.mu[k], self.sigma[k], allow_singular=True)
            sum_yp += yp
            plt.plot(x, y)
            plt.scatter(self.data, yp, marker='o', color="orange")

        plt.figure(1)
        plt.title('mixtures')
        plt.plot(x, sum_y, 'g-')
        plt.scatter(self.data, sum_yp, marker='o', color="red")

    # TODO: add normal pdf, do not use api
    def single_variable_gausian(self, x, mu, sigma):
        return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power((x - mu)/sigma, 2.)/2)

# if __name__ == '__main__':
# #     data = np.array([1,2,3,6,7,99,100,660,-101,-100,-3,0], dtype=np.float)
# #     # gmm = GMM(2, data)
# #     # gmm = GMM(2, data, init_method='uniform')
# #     gmm = GMM(3, data, init_method='uniform')
# #     gmm.train(100)
# #     a = gmm