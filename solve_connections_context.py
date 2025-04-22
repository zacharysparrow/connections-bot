#import time
import numpy as np
#import csv
import random
from itertools import combinations

import matplotlib.pyplot as plt

#from nltk.corpus import wordnet as wn
#from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import qr

from utils import *

large_width = 400
np.set_printoptions(linewidth=large_width)


puzzle = read_csv("puzzles/test_puzzle_3.txt") #3 is easy, 2 is medium, 1,4 are hard
puzzle_words = [x.lower().strip() for xs in puzzle for x in xs]
print(np.reshape(puzzle_words,(4,4)))
print()

#model = SentenceTransformer("all-MiniLM-L6-v2")
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
n_embeddings_per_word = 3


def is_allowed(i, cluster, wrong_groups, close_groups, ww2w, words):
    if i in cluster:
        return False
    potential_cluster = cluster + [i]
    for g in wrong_groups:
        check = sum(words[ww2w[j]] in g for j in potential_cluster)
        if check > 2: # check wrong groups
            return False
    for g in close_groups:
        check = sum(words[ww2w[j]] in g for j in potential_cluster)
        if not (check % 2 == 1): # check close groups
#            if check != 0:
            return False
    else:
        return True


def add_to_cluster(cluster, sim_mat, size, wrong_groups, close_groups, ww2w, words):
    n_to_add = size - len(cluster)
    best_sim = [0.0 for i in range(len(sim_mat))]
    for i,row in enumerate(sim_mat):
#        if i not in cluster:
        if is_allowed(i, cluster, wrong_groups, close_groups, ww2w, words):
            clust_sim = sum(row[cluster])
            row_copy = row.copy()
            row_copy[cluster] = -1.0
            curr_sim = sum(sorted(row_copy)[-n_to_add:])
            best_sim[i] = curr_sim + clust_sim

    best_addition = np.argmax(best_sim)
    return best_addition


def find_cluster(sim_mat, size, wrong_groups, close_groups, ww2w, words):
    cluster = []
    for i in range(size):
        cluster.append(add_to_cluster(cluster, sim_mat, size, wrong_groups, close_groups, ww2w, words))
        print(cluster)
    return cluster


def enumerate_groups(sim_mat):
    n_points = len(sim_mat)
    result = {}
    for i in combinations(range(n_points),4):
        result[i] = np.sum(sim_mat[np.ix_(i,i)])
    return result


def index_of_substring(tokens, substring, word_ids):
    #use word_ids to group tokens appropriately
    the_list = []
    stripped_tokens = [x.replace('Ġ','') for x in tokens if x not in ['[CLS]','[SEP]']]
    for i,w in enumerate(word_ids):
        if i < len(word_ids)-1:
            if w == word_ids[i+1]:
                tokens[i] = tokens[i]+ tokens[i+1]
                tokens.pop(i+1)
                word_ids.pop(i+1)
    substring_pos = []
    for i, s in enumerate(tokens):
        if substring in s:
              substring_pos.append(i)
    return substring_pos


def get_embeddings(words, model, tokenizer):
    token_input = tokenizer(" ".join(words), return_tensors="pt")
    word_pos = []
    for w in words:
        curr_pos = index_of_substring(tokenizer.convert_ids_to_tokens(token_input['input_ids'][0]), w, token_input.word_ids())
        word_pos.append(curr_pos)
    
    all_embeddings = model(**token_input)
    all_embeddings = all_embeddings['last_hidden_state'][0].detach().numpy()
    
    embeddings = []
    for wp in word_pos:
        embeddings.append(all_embeddings[wp][0])

    return embeddings


def select_clusters(words, n_defns):
    random.shuffle(words)
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
#    model.embeddings.tok_embeddings.weight.data = torch.zeros((50368, 768))

    embeddings = []
    copy_words = words.copy()
    for i in range(30):
#        random.shuffle(copy_words)
        shuffled_embeddings = np.array(get_embeddings(copy_words, model, tokenizer))
        order = []
        for og in words:
            for i,w in enumerate(copy_words):
                if w == og:
                    order.append(i)
        unshuffled_embeddings = shuffled_embeddings[order]
        embeddings.append(unshuffled_embeddings)
    embeddings = list(np.mean(np.array(embeddings), axis=0))

    sim_mat = cosine_similarity(embeddings)

    plt.matshow(sim_mat)
    plt.colorbar()
    plt.title("All Similarities")
    plt.show()

    all_groups = enumerate_groups(sim_mat)
    sorted_keys = sorted(all_groups, key=all_groups.get, reverse=True)

    groups = []
    wrong_groups = []
    close_groups = []
    lives_left = 4
    while lives_left > 0 and len(groups) < 4:
        if sorted_keys == []:
            raise Exception("I ran out of options!")
        connection = list(sorted_keys[0])
#        print(connection)
        found_words = [w for i,w in enumerate(words) if i in connection]
        print(found_words)
        if len(found_words) != 4:
            raise Exception("I messed up somewhere!")
        
        while True:
            user_input = input("Is this a valid connection? (y/n/3)")
            if user_input == 'y' or user_input == 'n' or user_input == '3':
                break
            else:
                print("Please enter a valid input. (y/n/3)")
                continue
        
        if user_input == 'y':
            groups.append(found_words)
            sorted_keys = [k for k in sorted_keys if not any(i in connection for i in k)]
        
        elif user_input == 'n':
            lives_left -= 1
            wrong_groups.append(found_words)
            sorted_keys = [k for k in sorted_keys if sum(i in connection for i in k) <= 2]
            if lives_left == 0:
                print()
                print("I found "+str(len(groups))+"/4 connections.")
                print("Better luck next time!")
                quit()
            print("Remaining tries: "+str(lives_left))
        
        elif user_input == '3':
            lives_left -= 1
            close_groups.append(found_words)
            sorted_keys = [k for k in sorted_keys if (sum(i in connection for i in k) in set([0,1,3]))]
            if lives_left == 0:
                print()
                print("I found "+str(len(groups))+"/4 connections.")
                print("Better luck next time!")
                quit()
            print("Remaining tries: "+str(lives_left))
        
        print()

    print()
    print("These are your connections!")
    print(np.array(groups))
    return(groups)

#best_guess = np.array(select_clusters(puzzle_words, model))
#puzzle_words.sort()
#print(puzzle_words)
test_words = puzzle_words.copy()
#test_words = puzzle_words[0:8].copy()
#test_words.sort()
#best_guess = np.array(select_clusters(test_words, model, n_embeddings_per_word))
best_guess = np.array(select_clusters(test_words, n_embeddings_per_word))
