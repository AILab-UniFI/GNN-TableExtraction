import os
import pickle
import random
from attrdict import AttrDict
from matplotlib import pyplot as plt
import numpy as np

from typing import List
from sklearn.manifold._t_sne import TSNE

from sklearn.mixture import GaussianMixture

from src.utils.fs import create_folder, create_folder_if_not_exists
from src.utils.matrixes import affinity
from src.utils.nums import to_numeral
from src.utils.paths import TABLES_REPR, TABLES_PREPROCESS
from src.utils.seeds import set_seeds
from src.utils.const import RANDOM_SEED

from src.components.tables.som import MiniSom as SOM, weighted_log
from src.components.tables.gmm import HardEMGaussianMixture

from src.components.tables.levenshtein import LevenshteinSimilitudes
from src.components.tables.extractor import Extractor
from src.components.tables.vocabulator import Vocabulator

from src.visualization.plots import plot_tsne, print_clusters

set_seeds(RANDOM_SEED)

class Preprocessor(object):
    
    extractor: Extractor
    vocabulator: Vocabulator
    tables_files: List[str]
    random_seed = RANDOM_SEED

    def __init__(self, unk='<UNK>', args:AttrDict = AttrDict({})):
        # window is in extractor.window
        # not in self.window = window 
        self.unk = unk

        self.nc = {}
        self.rc = {}
        self.wc = {self.unk: 1}
        self.prototypes = None

    def create_Extractor(self, Extractor: Extractor, name):
        self.extractor = Extractor(name=name, window=3, unk=self.unk)
        return self.extractor

    def create_Vocabulator(self, Vocabulator: Vocabulator, name):
        self.vocabulator = Vocabulator(name=name, unk=self.unk)
        return self.vocabulator

    def text_skipgram(sentence, i, window):
        """
        Skipgram representation for text
        Input:
            - :sentence: str
            - :i: center index
            - :window: number of element to the left and rigth of i
        Return:
            - :(iword, oword): iword, left + right
        """

        iword = sentence[i]
        left = sentence[i - window: i]
        right = sentence[i + 1: i + 1 + window]
        return iword, left + right

    def table_skipgram(extractor: Extractor, table_df, i, j, window=None):
        """
        Skipgram representation for tables
        Input:
            - :table_df: pd.DataFrame object N*M
            - :i: row center index
            - :j: column center index
            - :window: number of element from row-column to take into consideration
        Return:
            - :(iword, oword): iword, left + right
        """

        extractor = extractor.extract(table_df, i, j, window)
        return extractor.data


    def train_som(self, prototypes=10, sigma=0.03, lr=0.3, iters=10000, log_space=False):
        """
        :param nc_path: path under the save_directory
        :param prototypes: number of SOM neurons
        :param sigma:  sigma of SOM
        :param lr:  learning rate of SOM
        :return: None

        Train a simple SOM, and save it's neuron weights as prototypes, given numeral counts

        """
        nc = pickle.load(open(TABLES_PREPROCESS / 'nc.dat', 'rb'))

        # unfold and shuffle nc data
        data = []
        for k, v in nc.items():
            if to_numeral(k) == None:
                print('invalid numeral {}'.format(k))
            else:
                data += [[to_numeral(k)]]*v

        print('total number of different numerals: ', len(nc))
        print('total number of numeral samples: ', len(data))

        random.shuffle(data)
        if log_space:
            data = [[weighted_log(x[0])] for x in data]

        som = SOM(prototypes, 1, 1, sigma=sigma, learning_rate=lr, random_seed=RANDOM_SEED)  # initialization

        print("Training SOMs...")
        # som.random_weights_init(data)
        som.train_random(data, iters)  # trains the SOM with 1000 iterations
        print("...Ready!")
        # win_map = som.win_map(data)
        self.prototypes = som.get_weights().reshape(prototypes) # nd array
        if log_space:
            SOM_ = TABLES_REPR / 'som_log'
        else:
            SOM_ = TABLES_REPR / 'som'

        if not os.path.exists(SOM_):
            os.makedirs(SOM_)
        print('prototypes: \n{}'.format(self.prototypes))
        pickle.dump(self.prototypes, open(SOM_ / f'prototypes-{prototypes}-{sigma}-{lr}.dat', 'wb'))
        print('...Saving Prototypes')



    def train_gmm(self, components=20, iters=100, gmm_init_mode='rd', gmm_type='soft', prototype_path=None, log_space=False):

        # print('<<<<<<<<<<INITIALIZING>>>>>>>>>> \n means: {} \n sigma: {}\n, weights: {}'.format(gmm.means_, gmm.covariances_, gmm.weights_))
        assert gmm_init_mode in ['rd', 'fp', 'km']
        assert gmm_type in ['soft', 'hard']
        nc = pickle.load(open(TABLES_PREPROCESS / 'nc.dat', 'rb'))

        # we use fix random seed
        # random.seed(100)
        # unfold and shuffle nc data
        data = []
        for k, v in nc.items():
            if to_numeral(k) == None:
                print('invalid numeral {}'.format(k))
            else:
                data += [[to_numeral(k)]]*v

        print('total number of different numerals: ', len(nc))
        print('total number of numeral samples: ', len(data))

        # shuffle and subsample for MEM problem
        random.shuffle(data)

        if len(data) > 2000000:
            data = data[:2000000]

        if log_space:
            data = [weighted_log(x[0]) for x in data]

        print('subsampled to {}'.format(len(data)))

        data = np.array(data).reshape(-1,1)
        # getting initialization parameters
        if gmm_init_mode == 'km':

            if gmm_type == 'soft':
                gmm = GaussianMixture(components, max_iter=iters, n_init=1, verbose=10, init_params='kmeans')
            else:
                gmm = HardEMGaussianMixture(components, max_iter=iters, n_init=1, verbose=10, init_params='kmeans')


        else:
            # random select means
            if gmm_init_mode == 'rd' :
                prototypes = np.random.choice(data.reshape(-1), components)
            else:
                assert prototype_path is not None
                if log_space:
                    GMM_ = TABLES_REPR /  'gmm_log'
                else:
                    GMM_ = TABLES_REPR / 'gmm'

                prototypes = pickle.load(open(GMM_ / prototype_path, 'rb'))

                assert len(prototypes) == components

            mus = prototypes
            min_sigma = 1e-6

            diff = np.abs(data.reshape(len(data)) - mus[:, np.newaxis])

            amin = np.argmin(diff, axis=0)

            K = len(prototypes)
            clusters = [[0] for i in range(K)]
            for i in range(len(data)):
                clusters[amin[i]].append(data[i])

            means = np.array([np.mean(i) for i in clusters]).reshape(-1, 1)

            covs = np.array([np.std(i) if len(i) > 1 else min_sigma for i in clusters]).reshape(-1, 1, 1)
            precision = np.linalg.inv(covs)

            weights = np.array([len(c) for c in clusters])
            weights = weights / np.sum(weights)

            if gmm_type == 'soft':
                gmm = GaussianMixture(components, max_iter=iters, n_init=1, verbose=10, means_init=means,
                                  precisions_init=precision, weights_init=weights)
            else:
                gmm = HardEMGaussianMixture(components, max_iter=iters, n_init=1, verbose=10, means_init=means,
                                  precisions_init=precision, weights_init=weights)

        gmm.fit(data)
        if log_space:
            GMM_  = TABLES_REPR / 'gmm_log'
        else:
            GMM_  = TABLES_REPR / 'gmm'

        if not os.path.exists(GMM_):
            os.makedirs(GMM_)

        def single_variable_gaussian(x, mu, sigma):
            return 1. / (np.sqrt(2. * np.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.) / 2)

        def draw(gmm, X):
            x_min, x_max = min(X), max(X)
            # x = np.linspace(x_min, x_max, 10000)
            # x = np.array([])
            # for i in range(len(gmm.means_)):
            #     range_min, range_max = gmm.means_[i][0]-2 * gmm.covariances_[i][0], gmm.means_[i][0] + 2 * gmm.covariances_[i][0]
            #     x = np.append(x, np.linspace(range_min, range_max, 20))
            # x.sort()
            # print(x)
            print(gmm.means_)
            print(gmm.covariances_)
            print(gmm.weights_)

            X.sort()
            sum_y = np.zeros_like(X)
            plt.figure(0)
            plt.title('components')
            for i in range(len(gmm.means_)):
                y = single_variable_gaussian(X, gmm.means_[i][0], gmm.covariances_[i][0])
                y[y > 1] = 0 # set to 0 for better plot!
                sum_y += y * gmm.weights_[i]
                # yp = single_variable_gaussian(X, gmm.means_[i][0], gmm.covariances_[i][0])
                # yp[yp > 1] = 0
                # sum_yp += yp
                plt.plot(X, y)
            plt.savefig(GMM_ / f'components-{components}.png')

            plt.figure(1)
            plt.title('mixtures')

            plt.plot(X, sum_y, 'g-')
            plt.savefig(GMM_ / f'mixture-{components}.png')


        # 'rd' indicates for random initialization, 'fp' for 'from prototypes'

        pickle.dump(gmm, open(GMM_ / f'gmm-{components}-{gmm_init_mode}-{gmm_type}.dat'))
        print('means: {gmm.means_} \n sigma: {}\n, weights: {}'.format(gmm.means_, gmm.covariances_, gmm.weights_))

        if log_space:
            data_points = np.array([weighted_log(x) for x in np.array(list(nc.keys()), dtype=np.float32)]).reshape(-1,1)
        else:
            data_points = np.array(list(nc.keys()), dtype=np.float32).reshape(-1,1)

        posterior = gmm.predict_proba(data_points)
        path = GMM_ / f'gmm_posterior-{components}-{gmm_init_mode}-{gmm_type}.dat'
        pickle.dump(posterior, open(path, 'wb'))
        print('...Saving trained GMMs objects to {}'.format(path))


    def train_repr(self, prototypes=10, n_components=4, limit=2000, debug=False):
        """
        :param prototypes: number of SOM neurons
        :param limit:  number of representations from dictionary to consider
        :return: None

        Create representative prototypes for Representation clusters, and save them as prototypes, given representation counts

        """
        
        rc = pickle.load(open(TABLES_PREPROCESS / 'rc.dat', 'rb'))
        words = sorted(rc, key=rc.get, reverse=True)
        
        limit = len(words) if limit>len(words) else limit
        words = words[:limit]

        words = np.asarray(words)
        
        # ord_words = [tuple(ord(w) for w in word) for word in words]
        # print(ord_words)

        lev_sim = LevenshteinSimilitudes()
        lev_sim.init_weights()
        matrix = lev_sim.calculate_similarity(words)
        
        for idx in [1968, 1488, 1059, 801, 470]:
            print(words[idx])
            print(ord(words[idx]))

        assert matrix.shape == (limit,limit), 'error matrix shape'
        
        centers, labels = affinity(matrix)

        matrix = np.square(np.array(matrix))
        
        nanmin_ = np.nanmean(matrix[matrix != np.inf])
        nanmax_ = np.nanmax(matrix[matrix != np.inf])
        
        matrix = np.nan_to_num(matrix, nan=nanmin_, posinf=nanmax_, neginf=nanmax_)

        assert np.any(np.isnan(matrix)) == False, 'error matrix nan'
        assert np.all(np.isfinite(matrix)) == True, 'error matrix infinite'

        embeddings = TSNE(n_components=n_components, metric='precomputed').fit_transform(matrix)
        
        if debug: 
            print('Representations: ')
            print_clusters(centers, labels, words)
            plot_tsne(embeddings, labels)

        create_folder_if_not_exists(TABLES_PREPROCESS)
        assert os.path.exists(TABLES_PREPROCESS)
        
        embed_repr = {
            'centers': centers, 
            'labels': labels, 
            'words': words,
            'embeddings': embeddings
        }
        pickle.dump(embed_repr, open(TABLES_PREPROCESS / f'embed-repr-{len(centers)}-{n_components}-{limit}.dat', 'wb'))
        print('...Saving Prototypes')

    def get_sent(self, line):
        # originally it returns sentence with all numbers and words in vocab
        # TODO implement method
        raise Exception('Non implemented method!')
