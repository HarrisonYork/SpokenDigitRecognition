import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def read_blocks(datafile):
    blocks = []
    current_block = []

    with open(datafile, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line == "":
                if current_block:
                    blocks.append(np.array(current_block))
                    current_block = []
            else:
                numbers = list(map(float, line.split()))
                if len(numbers) == 13:
                    current_block.append(numbers)

    if current_block:
        blocks.append(np.array(current_block))

    return blocks


def calc_distance(frame, mean):
    dist = 0
    for i in range(len(mean)):
        dist += (frame[i] - mean[i])**2
    return dist


def calc_mean(cluster):
    new_mean = [0 for _ in range(13)]
    for frame in cluster:
        for mfcc in range(len(frame)):
            new_mean[mfcc] += frame[mfcc]
    for mfcc in range(len(new_mean)):
        new_mean[mfcc] = new_mean[mfcc] / len(cluster)
    return new_mean


def calc_covariance(cluster, mean, x, y):
    cov = 0
    x_var = 0
    y_var = 0
    for i in range(len(cluster)):
        cov += (cluster[i][x] - mean[x])*(cluster[i][y] - mean[y])
        x_var += (cluster[i][x] - mean[x])**2
        y_var += (cluster[i][y] - mean[y])**2
    cov = cov / len(cluster)
    x_var = x_var / len(cluster)
    y_var = y_var / len(cluster)
    return [[x_var, cov], [cov, y_var]]


# separate blocks corresponding to a single digit
file = 'Train_Arabic_Digit.txt'
allblocks = read_blocks(file)

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

num_clusters = 3

for digit in range(1):
    a = digit*660
    b = (digit+1)*660
    digit_blocks = allblocks[a:b]

    means = [[0 for _ in range(13)] for _ in range(num_clusters)]

    # do k means on single digit with 3 means / clusters -> phonemes
    max_iterations = 15
    for i in range(max_iterations):
        # reset phoneme clusters
        clusters = [[] for _ in range(num_clusters)]

        # calculate min distance from each frame to mean
        for block in digit_blocks:
            for frame in block:
                distances = [0 for _ in range(num_clusters)]
                for d in range(len(distances)):
                    distances[d] = calc_distance(frame, means[d])

                min_distance = distances[0]
                min_dist_index = 0
                for d in range(len(distances)):
                    if distances[d] < min_distance:
                        min_distance = distances[d]
                        min_dist_index = d

                clusters[min_dist_index].append(frame)

        # recalculate cluster mean after each iteration
        for m in range(len(means)):
            if len(clusters[m]) > 0:
                means[m] = calc_mean(clusters[m])


    # plot gaussians
    xpoints, ypoints = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    pos = np.dstack((xpoints, ypoints))

    Y_Axis = [0, 0, 1]
    X_Axis = [1, 2, 2]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    print(len(clusters))
    print(len(clusters[0]))
    print(len(clusters[0][0]))

    clusterpoints = [[] for _ in range(12)]

    for frame in clusters[0]:
        clusterpoints[0].append(frame[0])
        clusterpoints[1].append(frame[1])
        clusterpoints[2].append(frame[2])

    for frame in clusters[1]:
        clusterpoints[3].append(frame[0])
        clusterpoints[4].append(frame[1])
        clusterpoints[5].append(frame[2])

    for frame in clusters[2]:
        clusterpoints[6].append(frame[0])
        clusterpoints[7].append(frame[1])
        clusterpoints[8].append(frame[2])

    # for frame in clusters[3]:
    #     clusterpoints[9].append(frame[0])
    #     clusterpoints[10].append(frame[1])
    #     clusterpoints[11].append(frame[2])

    for i in range(3):
        x = X_Axis[i]
        y = Y_Axis[i]

        # calculate covariance
        covariance1 = calc_covariance(clusters[0], means[0], x, y)
        covariance2 = calc_covariance(clusters[1], means[1], x, y)
        covariance3 = calc_covariance(clusters[2], means[2], x, y)
        # covariance4 = calc_covariance(clusters[3], means[3], x, y)

        rv = multivariate_normal(mean=[means[0][x], means[0][y]], cov=covariance1)
        axs[i].contour(xpoints, ypoints, rv.pdf(pos), levels=5)
        rv = multivariate_normal(mean=[means[1][x], means[1][y]], cov=covariance2)
        axs[i].contour(xpoints, ypoints, rv.pdf(pos), levels=5)
        rv = multivariate_normal(mean=[means[2][x], means[2][y]], cov=covariance3)
        axs[i].contour(xpoints, ypoints, rv.pdf(pos), levels=5)
        # rv = multivariate_normal(mean=[means[3][x], means[3][y]], cov=covariance4)
        # axs[i].contour(xpoints, ypoints, rv.pdf(pos), levels=5)

        axs[i].scatter(clusterpoints[x], clusterpoints[y], color='red', alpha=0.2, s=1)
        axs[i].scatter(clusterpoints[x+3], clusterpoints[y+3], color='blue', alpha=0.2, s=1)
        axs[i].scatter(clusterpoints[x+6], clusterpoints[y+6], color='green', alpha=0.2, s=1)
        # axs[i].scatter(clusterpoints[x + 9], clusterpoints[y + 9], color='orange', alpha=0.2, s=1)

        axs[i].scatter(means[0][x], means[0][y], color='red', label='Cluster 1')
        axs[i].scatter(means[1][x], means[1][y], color='blue', label='Cluster 2')
        axs[i].scatter(means[2][x], means[2][y], color='green', label='Cluster 3')
        # axs[i].scatter(means[3][x], means[3][y], color='orange', label='Cluster 4')

        axs[i].scatter(means[0][x], means[0][y], color='black', label='Cluster Means')
        axs[i].scatter(means[1][x], means[1][y], color='black')
        axs[i].scatter(means[2][x], means[2][y], color='black')
        # axs[i].scatter(means[3][x], means[3][y], color='black')

        axs[i].set_title(f'Digit {digit}, MFCC {y+1} vs MFCC {x+1}', size=18)
        axs[i].set_xlabel(f'MFCC {x+1}', size=18)
        axs[i].set_ylabel(f'MFCC {y+1}', size=18)

        axs[i].legend()

    fig.tight_layout()

    plt.suptitle(f'K-Means GMM for Digit {digit} with {num_clusters} Clusters', size=20)
    plt.subplots_adjust(top=0.85)
    plt.show()
