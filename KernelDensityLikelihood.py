import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
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


def calc_covariance(cluster, mean):
    output = [[0 for _ in range(13)] for _ in range(13)]
    if (len(cluster) == 0):
        return output

    for i in range(13):
        for j in range(13):
            cov = 0
            for x in range(len(cluster)):
                cov += (cluster[x][i] - mean[i]) * (cluster[x][j] - mean[j])
            output[i][j] = cov / len(cluster)

    # for i in range(13):
    #     cov = 0
    #     for x in range(len(cluster)):
    #         cov += (cluster[x][i] - mean[i]) ** 2
    #     output[i][i] = cov / len(cluster)

    # all_ = 0
    # for i in range(13):
    #     for frame in cluster:
    #         all_ += (frame[i] - mean[i]) ** 2
    #
    # for i in range(13):
    #     output[i][i] = all_ / (13 * len(cluster))

    return output


def count(clusters):
    total = 0
    for c in clusters:
        total += len(c)
    return total


def probability(clusters, total):
    probabilities = [0 for _ in range(len(clusters))]
    for i in range(len(clusters)):
        if (len(clusters[i]) == 0):
            continue
        probabilities[i] = (len(clusters[i])/total)
    return probabilities


def calc_likelihood(x, mean, covariance, probability):
    n = probability * multivariate_normal.pdf(x, mean=mean, cov=covariance, allow_singular=True)
    return n


def get_likelihood(x, covariance, mean):
    n = multivariate_normal.pdf(x, mean=mean, cov=covariance, allow_singular=True)
    return n


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

# do k means on single digit with 3 means / clusters -> phonemes
max_iterations = 15
for i in range(max_iterations):
    # reset phoneme clusters
    clusters = [[] for _ in range(number_clusters)]

    # calculate min distance from each frame to mean
    for block in digit_blocks:
        for frame in block:
            distances = [0 for _ in range(number_clusters)]
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

number_points = count(clusters)
probabilities = probability(clusters, number_points)
covariances = [[[0 for _ in range(number_mfcc)] for _ in range(number_mfcc)] for _ in range(number_clusters)]
for i in range(len(clusters)):
    covariances[i] = calc_covariance(clusters[i], means[i])

all_digit_likelihoods = []

for i in range(10):
    digit2 = i
    color = 'blue'
    c = digit2*660
    d = (digit2+1)*660
    new_blocks = allblocks[c:d]

    digit_likelihood = []

    for utterance in new_blocks:
        utterance_likelihood = 0
        for frame in utterance:
            max_cluster_likelihood = float('-inf')
            for i in range(len(clusters)):
                cluster = clusters[i]
                covariance = covariances[i]
                mean = means[i]
                prob = probabilities[i]

                lik = np.log(get_likelihood(frame, covariance, mean))
                li = np.log(prob) + lik
                if (li > max_cluster_likelihood):
                    max_cluster_likelihood = li
            utterance_likelihood += max_cluster_likelihood
        digit_likelihood.append(utterance_likelihood)
    all_digit_likelihoods.append(digit_likelihood)

for i in range(10):
    dl = all_digit_likelihoods[i]
    digit_likelihood = np.reshape(dl, (-1, 1))
    kde = KernelDensity(kernel='gaussian', bandwidth=5.0).fit(digit_likelihood)

    x_vals = np.linspace(min(digit_likelihood), max(digit_likelihood), 1000)
    log_density = kde.score_samples(x_vals.reshape(-1, 1))

    plt.plot(x_vals, np.exp(log_density), label=f'Likelihood Digit {i}')
    plt.xlabel("Likelihood", size=18)
    plt.ylabel("Density", size=18)

str = f"Kernel Density Estimate of Likelihood with GMM {digit}, K-Means"
plt.legend()
plt.title(str, size=20)
plt.show()


