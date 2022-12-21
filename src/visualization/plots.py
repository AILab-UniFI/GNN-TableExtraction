import numpy as np

def print_clusters(centers, labels, words):
    unique_clusters = np.unique(labels)
    print(f'N° clusters: {len(unique_clusters)}')
    for cluster_id in unique_clusters:
        exemplar = words[centers[cluster_id]]
        cluster = np.unique(words[np.nonzero(labels==cluster_id)])
        cluster_str = ", ".join(cluster)
        print("C %s:   %s" % (exemplar.ljust(15), cluster_str))

def plot_tsne(Y, color):
    import matplotlib
    matplotlib.use('agg')

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import NullFormatter

    # Next line to silence pyflakes. This import is needed.
    Axes3D

    # Create figure
    fig = plt.figure(figsize=(15, 8))
    # Add 3d scatter plot
    ax = plt.axes()
    

    # Plot results
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_title("tsne plot")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")

    plt.savefig('tsne-plot')