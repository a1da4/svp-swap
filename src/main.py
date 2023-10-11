import os
import argparse
import pickle
import logging
import random
import torch
import numpy as np
from scipy.stats import spearmanr

from utils import *
from svpgauss.src.utils import *


def main_shuffle(args):
    os.makedirs("../results", exist_ok=True)
    logging.basicConfig(filename="../results/main_shuffle.log", format="%(asctime)s %(message)s", level=logging.INFO)
    logging.info(f"[main_shuffle] args: {args}")

    logging.info("[main_shuffle] 1. load word2grade (gold) ...")
    word2gold = {}
    with open(args.graded_words_list) as fp:
        for line in fp:
            word, grade = line.strip().split("\t")
            word2gold[word] = float(grade)


    logging.info("[main_shuffle] 2. load word2vecs / obtain word2gauss...")
    word2vecs_c1 = load_word2vecs(args.wordvec_pathes[0])
    word2gauss_c1 = obtain_word2gauss(word2vecs_c1)
    word2vecs_c2 = load_word2vecs(args.wordvec_pathes[1])
    word2gauss_c2 = obtain_word2gauss(word2vecs_c2)
    logging.info("[main_shuffle] word2gauss obtained successfully")


    if args.pathes_corpora is not None: 
        logging.info("[main_shuffle] X. count total number of words in corpora c1/c2...")
        freq_normalize = True
        freq_c1 = count_corpus(args.pathes_corpora[0])
        freq_c2 = count_corpus(args.pathes_corpora[1])
        logging.info("[main_shuffle] freq_c1/c2 obtained successfully")
    else:
        freq_normalize = False
        freq_c1 = None
        freq_c2 = None


    logging.info("[main_shuffle] 3. predict with genuine C1, C2")
    metrics_diag = ["kl_c1_c2", "kl_c2_c1", "jeff",
               "l2_mean", "braycurtis_mean", "canberra_mean", 
               "chebyshev_mean", "cityblock_mean", "correlation_mean", "cosine_mean", 
               "l2_mean_cov", "braycurtis_mean_cov", "canberra_mean_cov",
               "chebyshev_mean_cov", "cityblock_mean_cov", "correlation_mean_cov", "cosine_mean_cov"]
    word2pred_diag = calculate_metrics(word2gold.keys(), word2gauss_c1, word2gauss_c2, metrics_diag, cov_component="diag")

    logging.info("[main_shuffle] 4. predict with shuffled C1, C2")
    #shuffle_func = "distance"
    shuffle_func = "random"
    rate_seed2spearman_diag = {}
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    seeds = [1, 19, 304, 1527, 12345, 
             121467, 293011, 1036694, 3977767, 7634694,
             4101133, 11447289, 28233559, 40732683, 89869379,
             234990455, 460513751, 674181274, 1965657259, 3383725953]

    for k in rates:
        rate_seed2spearman_diag[k] = {}
        for seed in seeds:
            rate_seed2spearman_diag[k][seed] = {}
            logging.info(f"[main_shuffle]  - shuffle rate: {k}, seed: {seed}")
            word2gauss_shuffled_c1, word2gauss_shuffled_c2 = obtain_shuffled_word2gauss(word2vecs_c1, word2vecs_c2, k, seed, shuffle_func=shuffle_func, freq_normalize=freq_normalize, freq_c1=freq_c1, freq_c2=freq_c2) 

            logging.info(f"[main_shuffle]    - make predictions (diag, full)")
            word2pred_shuffled_diag = calculate_metrics(word2gold.keys(), word2gauss_shuffled_c1, word2gauss_shuffled_c2, metrics_diag, cov_component="diag")
        
            logging.info("[main_shuffle]    - calculate distance (diag, full)")
            word2dist_diag = calculate_distance_genuine_shuffle(word2pred_diag, word2pred_shuffled_diag, metrics_diag)

            logging.info("[main_shuffle]    - save results")
            write_results(word2gold, word2dist_diag, metrics_diag, f"{args.output_name}_shufflerate-{k}_seed-{seed}_diag")

            for metric_name in metrics_diag:
                list_gold = []
                list_pred = []
                for word in word2gold.keys():
                    gold = word2gold[word]
                    pred = word2dist_diag[word][metric_name]
                    list_gold.append(gold)
                    list_pred.append(pred)
                spearman = spearmanr(list_gold, list_pred).statistic
                rate_seed2spearman_diag[k][seed][metric_name] = spearman

            if shuffle_func == "distance":
                break

    write_spearman_allrates_allseeds(rate_seed2spearman_diag, rates, seeds, metrics_diag, args.output_name)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--wordvec_pathes", nargs=2, help="path of word2vecs (.pkl, obtained from bert")
    parser.add_argument("-l", "--graded_words_list", help="annotated target word list")
    parser.add_argument("-f", "--pathes_corpora", nargs=2, help="pathes of target corpora (freq_normalize in shuffle())")
    parser.add_argument("-o", "--output_name")
    args = parser.parse_args()
    main_shuffle(args)


if __name__ == "__main__":
    cli_main()
