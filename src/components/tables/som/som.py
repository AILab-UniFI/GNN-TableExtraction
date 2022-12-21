

from math import sqrt

from numpy import (unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot,
                   logical_and, std,)
from collections import defaultdict, Counter
from warnings import warn

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return sqrt(dot(x, x.T))


def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.
    t : int
        current iteration.
    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1+t/(max_iter/2))


class MiniSom(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 decay_function=asymptotic_decay,
                 neighborhood_function='gaussian', random_seed=None):
        """Initializes a Self Organizing Maps.
        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.
        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well.
        Parameters
        ----------
        x : int
            x dimension of the SOM.
        y : int
            y dimension of the SOM.
        input_len : int
            Number of the elements of the vectors in input.
        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
            learning_rate, initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)
        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            the default function is:
                        learning_rate / (1+t/(max_iterarations/2))
            A custom decay function will need to to take in input
            three parameters in the following order:
            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed
            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.
        neighborhood_function : function, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map
            possible values: 'gaussian', 'mexican_hat', 'bubble'
        random_seed : int, optional (default=None)
            Random seed to use.
        """
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = random.RandomState(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        # random initialization
        self._weights = self._random_generator.rand(x, y, input_len)*2-1

        # for i in range(x):
        #     for j in range(y):
        #         # normalization
        #         norm = fast_norm(self._weights[i, j])
        #         self._weights[i, j] = self._weights[i, j] / norm

        self._activation_map = zeros((x, y))
        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function
        self._decay_function = decay_function

        ## used to check terminal situation
        self._weight_diff_list = self._random_generator.rand(10,) * 1000
        self._exit_criteria = 0.07 #quant error std threshold
        # TODO may don't need?

        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and divmod(sigma, 1)[1] != 0:
            warn('sigma should be an integer when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

    def get_weights(self):
        """Returns the weights of the neural network"""
        return self._weights

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x"""
        s = subtract(x, self._weights)  # x - w
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            # || x - w ||
            self._activation_map[it.multi_index] = fast_norm(s[it.multi_index])
            it.iternext()

    def activate(self, x):
        """Returns the activation map to x"""
        self._activate(x)
        return self._activation_map

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c"""
        d = 2*pi*sigma*sigma
        ax = exp(-power(self._neigx-c[0], 2)/d)
        ay = exp(-power(self._neigy-c[1], 2)/d)
        return outer(ax, ay)  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c"""
        xx, yy = meshgrid(self._neigx, self._neigy)
        p = power(xx-c[0], 2) + power(yy-c[1], 2)
        d = 2*pi*sigma*sigma
        return exp(-p/d)*(1-2/d*p)

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
           sigma should be an odd value,
        """
        ax = logical_and(self._neigx > c[0]-sigma,
                         self._neigx < c[0]+sigma)
        ay = logical_and(self._neigy > c[1]-sigma,
                         self._neigy < c[1]+sigma)
        return outer(ax, ay)*1.

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x, find the BMU"""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn
        win : tuple
            Position of the winning neuron for x (array or tuple)
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations
        """
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig)*eta

        it = nditer(g, flags=['multi_index'])

        while not it.finished:
            # eta * neighborhood_function * (x-w)
            x_w = (x - self._weights[it.multi_index])
            self._weights[it.multi_index] += g[it.multi_index] * x_w
            # do not normalization
            # norm = fast_norm(self._weights[it.multi_index])
            # self._weights[it.multi_index] = self._weights[it.multi_index]/norm
            it.iternext()

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data)
        q = zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self._weights[self.winner(x)]
        return q

    def random_weights_init(self, data):
        """Initializes the weights of the SOM
        picking random samples from data"""
        self._check_input_len(data)
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self._weights[it.multi_index] = data[rand_i]
            # norm = fast_norm(self._weights[it.multi_index])
            # self._weights[it.multi_index] = self._weights[it.multi_index]/norm
            it.iternext()

    def weights_diff_norm(self,old,new):
        return linalg.norm(old-new)

    def train_random(self, data, num_iteration, varbose=True):
        """Trains the SOM picking samples at random from data"""
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        index = 0

        # initialize
        current_std = float('inf')
        diff = float('inf')

        for iteration in range(num_iteration):
            # pick a random sample
            rand_i = self._random_generator.randint(len(data))
            old_weights=self._weights.copy()
            self.update(data[rand_i], self.winner(data[rand_i]),
                        iteration, num_iteration)
            if iteration % 100 == 0:
                diff = self.weights_diff_norm(old_weights,self._weights)
                self._weight_diff_list[index] = diff
                index += 1
                if index == 10:
                    index = 0
                current_std = std(self._weight_diff_list)
            # if current_std < self._exit_criteria:
            #     print("diff: {}, diff std: {}, iteration:{}".format(diff,current_std,iteration))
            #     break
                # pass
            if varbose and (iteration % 1000 == 0):
                print("diff: {}, diff std: {}, iteration:{}".format(diff,current_std,iteration))

    def train_batch(self, data, num_iteration, varbose=True):
        """Trains using all the vectors in data sequentially"""
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        iteration = 0

        while iteration < num_iteration:
            idx = iteration % (len(data)-1)
            self.update(data[idx], self.winner(data[idx]),
                        iteration, num_iteration)
            iteration += 1

            if varbose and (iteration % 50 == 0):
                error = self.quantization_error(data)
                print('iteration: {}'.format(iteration), error)

    def distance_map(self):
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours."""
        um = zeros((self._weights.shape[0], self._weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if (ii >= 0 and ii < self._weights.shape[0] and
                            jj >= 0 and jj < self._weights.shape[1]):
                        w_1 = self._weights[ii, jj, :]
                        w_2 = self._weights[it.multi_index]
                        um[it.multi_index] += fast_norm(w_1-w_2)
            it.iternext()
        um = um/um.max()
        return um

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)
        error = 0
        max = 10000

        for x in data[:max]:
            error += fast_norm(x-self._weights[self.winner(x)])
        return error/len(data)

    def win_map(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j."""
        self._check_input_len(data)
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap

    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
        that contains the number of samples from a given label
        that have been mapped in position i,j.
        Parameters
        ----------
        data : data matrix
        label : list or array that contains the label of each sample in data.
        """
        self._check_input_len(data)
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.winner(x)].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap
