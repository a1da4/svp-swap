import logging
import random
import numpy as np
from tqdm import tqdm

from svpgauss.src.metrics import l2_norm
from svpgauss.src.utils import *


def count_corpus(path_corpus):
    """ count total number of words/sentences (for freq_norm in shuffle())

    :param path_corpus: path of corpus

    :return: freq_corpus, int, total number of words/sentences in the target corpus
    """
    logging.debug(f"[count_corpus] path: {path_corpus}")
    freq_corpus = 0
    with open(path_corpus) as fp:
        for line in fp:
            #words = line.strip().split()
            #freq_corpus += len(words)
            freq_corpus += 1

    logging.debug(f"[count_corpus] freq_corpus: {freq_corpus}")

    return freq_corpus


def obtain_shuffle_ids_random(num_vecs_c1, num_vecs_c2, num_shuffle, seed):
    """ sample shuffle ids randomly

    :param num_vecs_c1, num_vecs_c2: int, number of vectors
    :param num_shuffle: int, number of shuffle ids
    :param seed: int, random seed

    :return: ids_shuffle_c1, ids_shuffle_c2
    """
    np.random.seed(seed)
    random.seed(seed)

    ids_shuffle_c1 = random.sample(range(num_vecs_c1), num_shuffle)
    ids_shuffle_c2 = random.sample(range(num_vecs_c2), num_shuffle)
    logging.debug(f"[obtain_shuffle_ids_random] ids_shuffle_c1: {ids_shuffle_c1}")
    logging.debug(f"[obtain_shuffle_ids_random] ids_shuffle_c2: {ids_shuffle_c2}")

    return ids_shuffle_c1, ids_shuffle_c2


def obtain_shuffle_ids_distance(vecs_c1, vecs_c2, num_shuffle):
    """ sample shuffle ids from farther distance

    :param vecs_c1, vecs_c2: np.array, list of vectors
    :param num_shuffle: int, number of shuffle ids

    :return: ids_shuffle_c1, ids_shuffle_c2
    """
    mean_c1 = np.average(vecs_c1, axis=0)
    mean_c2 = np.average(vecs_c2, axis=0)
    logging.debug(f"[obtain_shuffle_ids_distance] mean_c1: {mean_c1.shape}")
    logging.debug(f"[obtain_shuffle_ids_distance] mean_c2: {mean_c2.shape}")

    distances_mean_c2_against_vecs_c1 = np.zeros([len(vecs_c1)])
    distances_mean_c1_against_vecs_c2 = np.zeros([len(vecs_c2)])
    logging.debug(f"[obtain_shuffle_ids_distance] distances (mean_c2 vs vecs_c1: {distances_mean_c2_against_vecs_c1.shape}")
    logging.debug(f"[obtain_shuffle_ids_distance] distances (mean_c1 vs vecs_c2: {distances_mean_c1_against_vecs_c2.shape}")

    for id_vecs_c1 in range(len(vecs_c1)):
        vec_each_c1 = vecs_c1[id_vecs_c1]
        distance = l2_norm(mean_c2, vec_each_c1)
        distances_mean_c2_against_vecs_c1[id_vecs_c1] += distance

    for id_vecs_c2 in range(len(vecs_c2)):
        vec_each_c2 = vecs_c2[id_vecs_c2]
        distance = l2_norm(mean_c1, vec_each_c2)
        distances_mean_c1_against_vecs_c2[id_vecs_c2] += distance

    ids_shuffle_c1 = np.argsort(distances_mean_c2_against_vecs_c1 * -1)[:num_shuffle]
    ids_shuffle_c2 = np.argsort(distances_mean_c1_against_vecs_c2 * -1)[:num_shuffle]
    logging.debug(f"[obtain_shuffle_ids_distance] ids_shuffle_c1: {ids_shuffle_c1}")
    logging.debug(f"[obtain_shuffle_ids_distance]  - top5 distances: {[distances_mean_c2_against_vecs_c1[id] for id in ids_shuffle_c1[:5]]}")
    logging.debug(f"[obtain_shuffle_ids_distance] ids_shuffle_c2: {ids_shuffle_c2}")
    logging.debug(f"[obtain_shuffle_ids_distance]  - top5 distances: {[distances_mean_c1_against_vecs_c2[id] for id in ids_shuffle_c2[:5]]}")

    return ids_shuffle_c1, ids_shuffle_c2


def shuffle(vecs_c1, vecs_c2, k, seed=1, shuffle_func="random", freq_normalize=False, freq_c1=None, freq_c2=None):
    """ shuffle vectors

    :param vecs_c1, vecs_c2: np.array, list of vectors
    :param k: float, shuffle rate (0 <= k <= 1)
    :param seed: int, random seed
    :param shuffle_func: str, name of metric ['random', 'distance']
    :param freq_normalize: bool, normalize frequency or not (freq_c1 * (freq_c1 / freq_c2))
    :param freq_c1, freq_c2: int, total number of words in corpus c1/c2

    :return: vecs_shuffled_c1, vecs_shuffled_c2
    """
    vecs_shuffled_c1 = np.zeros(vecs_c1.shape)
    vecs_shuffled_c2 = np.zeros(vecs_c2.shape)

    num_vecs_c1 = len(vecs_c1)
    num_vecs_c2 = len(vecs_c2)
    logging.debug(f"[shuffle] num_vecs_c1: {num_vecs_c1}")
    logging.debug(f"[shuffle] num_vecs_c2: {num_vecs_c2}")

    if freq_normalize:
        racio_c1_c2 = freq_c1 / freq_c2
        num_shuffle = round(min(num_vecs_c1, num_vecs_c2 * racio_c1_c2) * k)
    else:
        num_shuffle = round(min(num_vecs_c1, num_vecs_c2) * k)
    logging.debug(f"[shuffle] num_shuffle: {num_shuffle}")

    if shuffle_func == "random":
        ids_shuffle_c1, ids_shuffle_c2 = obtain_shuffle_ids_random(num_vecs_c1, num_vecs_c2, num_shuffle, seed)
    if shuffle_func == "distance":
        ids_shuffle_c1, ids_shuffle_c2 = obtain_shuffle_ids_distance(vecs_c1, vecs_c2, num_shuffle)

    num_shuffled = 0
    for id_vecs_c1 in range(num_vecs_c1):
        if id_vecs_c1 in set(ids_shuffle_c1):
            id_shuffle_c2 = ids_shuffle_c2[num_shuffled]
            vecs_shuffled_c1[id_vecs_c1] += vecs_c2[id_shuffle_c2]
            num_shuffled += 1
        else:
            vecs_shuffled_c1[id_vecs_c1] += vecs_c1[id_vecs_c1]

    num_shuffled = 0
    for id_vecs_c2 in range(num_vecs_c2):
        if id_vecs_c2 in set(ids_shuffle_c2):
            id_shuffle_c1 = ids_shuffle_c1[num_shuffled]
            vecs_shuffled_c2[id_vecs_c2] += vecs_c1[id_shuffle_c1]
            num_shuffled += 1
        else:
            vecs_shuffled_c2[id_vecs_c2] += vecs_c2[id_vecs_c2]
    
    return vecs_shuffled_c1, vecs_shuffled_c2


def obtain_shuffled_word2gauss(word2vecs_c1, word2vecs_c2, k, seed=1, shuffle_func="random", freq_normalize=False, freq_c1=None, freq_c2=None):
    """
    obtain dict[word] = Gauss

    :param word2vecs_c1, word2vecs_c2: dict[word] = array(len(usages), bert_dim)
    :param k: float, shuffle rate (0 <= k <= 1)
    :param seed: int, random seed
    :param shuffle_func: str, name of metric ['random', 'distance']
    :param freq_normalize: bool, normalize frequency or not (freq_c1 * (freq_c1 / freq_c2))
    :param freq_c1, freq_c2: int, total number of words in corpus c1/c2

    :return: word2gauss_shuffled_c1, word2gauss_shuffled_c2
    """
    word2gauss_shuffled_c1 = {}
    word2gauss_shuffled_c2 = {}
    for word in tqdm(word2vecs_c1.keys(), desc="[obtain shuffled word2gauss]"):
        vecs_c1 = word2vecs_c1[word]
        vecs_c2 = word2vecs_c2[word]
        vecs_shuffled_c1, vecs_shuffled_c2 = shuffle(vecs_c1, vecs_c2, k, seed, shuffle_func, freq_normalize, freq_c1, freq_c2)
        word2gauss_shuffled_c1[word] = Gauss(vecs_shuffled_c1)
        word2gauss_shuffled_c2[word] = Gauss(vecs_shuffled_c2)

    return word2gauss_shuffled_c1, word2gauss_shuffled_c2


def calculate_distance_genuine_shuffle(word2pred_genuine, word2pred_shuffled, metrics):
    """ calculate distance (word2pred_genuine, word2pred_shuffled)

    :param word2pred_genuine, word2pred_shuffled: dict, word:metric:value
    :param metrics: list of metrics

    :return: word2dist
    """
    word2dist = {} 
    for word in tqdm(word2pred_genuine.keys(), desc="[calculate distance (genuine, shuffled)]"):
        word2dist[word] = {}
        for metric in metrics:
            value_genuine = word2pred_genuine[word][metric]
            value_shuffled = word2pred_shuffled[word][metric]
            dist = l2_norm(value_genuine, value_shuffled)
            word2dist[word][metric] = dist

    return word2dist


def write_spearman_allrates_allseeds(rate_seed2spearman, rates, seeds, metrics, output_name):
    """
    save results (spearman r)

    :param rate_seed2spearman: dict[rate][seed]: {"METRIC": spearman, ...} 
    :param rates, seeds, metrics: list of rate / seed / metric name
    :param output_name: name of model / experiment
    """
    with open(f"../results/spearman_{output_name}.txt", "w") as fp:
        fp.write("rate\tseed")
        for metric in metrics:
            fp.write(f"\t{metric}")
        fp.write("\n")
        
        for rate in rate_seed2spearman.keys():
            for seed in rate_seed2spearman[rate].keys():
                fp.write(f"{rate}\t{seed}")
                for metric in metrics:
                    spearman = rate_seed2spearman[rate][seed][metric]
                    fp.write(f"\t{spearman}")
                fp.write("\n")

    with open(f"../results/spearman_mean_std_{output_name}.txt", "w") as fp:
        fp.write("rate")
        for metric in metrics:
            fp.write(f"\tmean({metric})\tstd({metric})")
        fp.write("\n")

        for rate in rate_seed2spearman.keys():
            fp.write(f"{rate}")
            for metric in metrics:
                spearmans_fixedrate_fixedmetric = []
                for seed in rate_seed2spearman[rate].keys():
                    spearman = rate_seed2spearman[rate][seed][metric]
                    spearmans_fixedrate_fixedmetric.append(spearman)

                spearman_mean = np.mean(spearmans_fixedrate_fixedmetric)
                spearman_std = np.std(spearmans_fixedrate_fixedmetric)
                fp.write(f"\t{spearman_mean}\t{spearman_std}")
            fp.write("\n")
