import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random

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


def calc_covariance(cluster, mean):
    output = [[0 for _ in range(13)] for _ in range(13)]
    if (len(cluster) == 0):
        return output

    # for i in range(13):
    #     for j in range(13):
    #         cov = 0
    #         for x in range(len(cluster)):
    #             cov += (cluster[x][i] - mean[i]) * (cluster[x][j] - mean[j])
    #         output[i][j] = cov / len(cluster)

    # for i in range(13):
    #     cov = 0
    #     for frame in cluster:
    #         cov += (frame[i] - mean[i])**2
    #     output[i][i] = cov / len(cluster)

    all_ = 0
    for i in range(13):
        for frame in cluster:
            all_ += (frame[i] - mean[i])**2

    for i in range(13):
        output[i][i] = all_ / (13 * len(cluster))

    return output


def calc_tied_covariance(clusters, means):
    output = [[0 for _ in range(13)] for _ in range(13)]

    count = 0
    for c in range(len(clusters)):
        cluster = clusters[c]
        mean = means[c]
        count += len(cluster)
        if (len(clusters) == 0):
            return output
        for i in range(13):
            for j in range(13):
                for frame in cluster:
                    output[i][j] += (frame[i] - mean[i]) * (frame[j] - mean[j])

    for i in range(13):
        for j in range(13):
            output[i][j] = output[i][j] / count

    # count = 0
    # for c in range(len(clusters)):
    #     cluster = clusters[c]
    #     mean = means[c]
    #     count += len(cluster)
    #     if len(clusters) == 0:
    #         return output
    #
    #     for frame in cluster:
    #         for i in range(13):
    #             output[i][i] += (frame[i] - mean[i]) ** 2
    #
    # for i in range(13):
    #     output[i][i] = output[i][i] / count

    # count = 0
    # all_ = 0
    # for c in range(len(clusters)):
    #     cluster = clusters[c]
    #     mean = means[c]
    #     count += len(cluster)
    #     if (len(clusters) == 0):
    #         return output
    #
    #     for frame in cluster:
    #         for i in range(13):
    #             all_ += (frame[i] - mean[i]) ** 2
    #
    # for i in range(13):
    #     output[i][i] = all_ / (13 * count)

    return output


def calc_probability(clusters):
    total = 0
    for c in clusters:
        total += len(c)

    probabilities = [0 for _ in range(len(clusters))]
    for i in range(len(clusters)):
        if (len(clusters[i]) == 0):
            continue
        probabilities[i] = (len(clusters[i])/total)
    return probabilities


def calc_likelihood(x, mean, covariance, probability):
    n = probability * multivariate_normal.pdf(x, mean=mean, cov=covariance, allow_singular=True)
    return n


def reduce_covariance(x, y, covariance):
    output = [[covariance[x][x], covariance[x][y]], [covariance[x][y], covariance[y][y]]]
    return output


# separate blocks corresponding to a single digit
file = 'Train_Arabic_Digit.txt'
allblocks = read_blocks(file)

digit = 0
number_clusters = 3
number_mfcc = 13

a = digit*660
b = (digit+1)*660
digit_blocks = allblocks[a:b]

means = [[random.uniform(-2, 2) for _ in range(number_mfcc)] for _ in range(number_clusters)]
# covariances is a list of 2D arrays corresponding to cluster covariance
covariances = [np.identity(13) for _ in range(number_clusters)]
probabilities = [0.5 for _ in range(number_clusters)]

# do expectation maximization on single digit with 3 means / clusters -> phonemes
max_iterations = 10
for i in range(max_iterations):
    # total frames per digit
    count_frames = 0

    # reset phoneme clusters
    clusters = [[] for _ in range(number_clusters)]

    for block in digit_blocks:
        for frame in block:
            count_frames += 1
            # calculate likelihood of a digit being in each cluster
            likelihoods = [0 for _ in range(number_clusters)]
            for l in range(len(likelihoods)):
                likelihoods[l] = calc_likelihood(frame, means[l], covariances[l], probabilities[l])

            max_likelihood = likelihoods[0]
            max_likelihood_index = 0
            for l in range(len(likelihoods)):
                if likelihoods[l] > max_likelihood:
                    max_likelihood = likelihoods[l]
                    max_likelihood_index = l

            clusters[max_likelihood_index].append(frame)

    # recalculate cluster mean after each iteration
    for m in range(len(means)):
        if len(clusters[m]) > 0:
            means[m] = calc_mean(clusters[m])

    # recalculate cluster covariance after each iteration
    for c in range(len(covariances)):
        if len(clusters[c]) > 0:
            covariances[c] = calc_covariance(clusters[c], means[c])
    # covar = calc_tied_covariance(clusters, means)
    # for c in range(len(covariances)):
    #     covariances[c] = covar

    # recalculate probability after each iteration
    # probability is len(cluster) / total frames per digit
    probabilities = calc_probability(clusters)

MFCCs = [[] for _ in range(number_clusters)]
MFCC1 = []
MFCC2 = []
MFCC3 = []
for block in digit_blocks:
    for frame in block:
        MFCC1.append(frame[0])
        MFCC2.append(frame[1])
        MFCC3.append(frame[2])

MFCCs[0] = MFCC1
MFCCs[1] = MFCC2
MFCCs[2] = MFCC3

xpoints, ypoints = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
pos = np.dstack((xpoints, ypoints))

Y_Axis = [0, 0, 1]
X_Axis = [1, 2, 2]

fig, axs = plt.subplots(1, 3, figsize=(16, 5))

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
    covariance1 = reduce_covariance(x, y, covariances[0])
    covariance2 = reduce_covariance(x, y, covariances[1])
    covariance3 = reduce_covariance(x, y, covariances[2])

    # rv = multivariate_normal(mean=[means[0][x], means[0][y]], cov=covariance1)
    # axs[i].contour(xpoints, ypoints, rv.pdf(pos), levels=5)
    # rv = multivariate_normal(mean=[means[1][x], means[1][y]], cov=covariance2)
    # axs[i].contour(xpoints, ypoints, rv.pdf(pos), levels=5)
    # rv = multivariate_normal(mean=[means[2][x], means[2][y]], cov=covariance3)
    # axs[i].contour(xpoints, ypoints, rv.pdf(pos), levels=5)

    axs[i].scatter(clusterpoints[x], clusterpoints[y], color='red', alpha=0.2, s=1)
    axs[i].scatter(clusterpoints[x + 3], clusterpoints[y + 3], color='blue', alpha=0.2, s=1)
    axs[i].scatter(clusterpoints[x + 6], clusterpoints[y + 6], color='green', alpha=0.2, s=1)

    axs[i].scatter(means[0][x], means[0][y], color='red', label='Cluster 1')
    axs[i].scatter(means[1][x], means[1][y], color='blue', label='Cluster 2')
    axs[i].scatter(means[2][x], means[2][y], color='green', label='Cluster 3')

    axs[i].scatter(means[0][x], means[0][y], color='black', label='Cluster Means')
    axs[i].scatter(means[1][x], means[1][y], color='black')
    axs[i].scatter(means[2][x], means[2][y], color='black')

    axs[i].set_title(f'Digit {digit}, MFCC {y + 1} vs MFCC {x + 1}', size=18)
    axs[i].set_xlabel(f'MFCC {x + 1}', size=18)
    axs[i].set_ylabel(f'MFCC {y + 1}', size=18)

    axs[i].legend()

fig.tight_layout()

plt.suptitle(f'Expectation-Maximization Clustering for Digit {digit} with Distinct Spherical Covariance', size=20)
plt.subplots_adjust(top=0.85)
plt.show()


