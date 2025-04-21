#import time
import numpy as np
#import csv
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


def make_sim_mat(embeddings, words_to_embeddings):
    sim_mat = cosine_similarity(embeddings)
    low_idx = 0
    high_idx = 0
    for i,e in enumerate(words_to_embeddings):
        n_embeddings = len(e)
        new_diag = np.eye(n_embeddings) - np.ones((n_embeddings,n_embeddings))
        high_idx += n_embeddings
        sim_mat[low_idx:high_idx,low_idx:high_idx] = new_diag
        low_idx = high_idx
    return sim_mat


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


def get_embeddings(word, n_defns, model, tokenizer):
    print(word)
    quotes = get_quotes(word) #need to lemmatize in here 

    inputs = []
    for i,q in enumerate(quotes):
        quotes[i] = q.replace('”', '').replace('“','').replace('"','')
        quote_input = tokenizer(quotes[i], return_tensors="pt", padding=True, truncation=True)#, is_split_into_words=True)
        inputs.append(quote_input)

    word_pos = []
    for i in inputs:
        curr_pos = index_of_substring(tokenizer.convert_ids_to_tokens(i['input_ids'][0]), word, i.word_ids())
        word_pos.append(curr_pos)
    
    all_embeddings = []
    for i in inputs:
        all_embeddings.append(model(**i)['last_hidden_state'].detach().numpy())

    embeddings = []
    for i,e in enumerate(all_embeddings):
        embeddings.append(e[0][word_pos[i]])

    embeddings = [item for sublist in embeddings for item in sublist]
    print(len(embeddings))

    _, _, qr_selection = qr(np.array(embeddings).T, pivoting=True)
    qr_selection = qr_selection[0:n_defns]
    embeddings = [embeddings[i] for i in qr_selection]
    return embeddings


def select_clusters(words, n_defns):
    model_id = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    embeddings = []
    for w in words:
        word_embedding = get_embeddings(w, n_defns, model, tokenizer)
        embeddings.append(word_embedding)
    embeddings_to_words = [i for i,w in enumerate(words) for j in embeddings[i]]
    words_to_embeddings = [[] for i in range(len(words))]
    counter = 0
    for i,w in enumerate(words):
        for e in embeddings[i]:
            words_to_embeddings[i].append(counter)
            counter += 1
    embeddings = [item for sublist in embeddings for item in sublist]
    print(len(embeddings))
    sim_mat = make_sim_mat(embeddings, words_to_embeddings)

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
        opt_solution = list(sorted_keys[0])
#        print(opt_solution)
        connection = [embeddings_to_words[i] for i in opt_solution]
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
        
        if user_input == 'y': #something not updated correctly here?
            embeddings_to_remove = [words_to_embeddings[i] for i in connection]
            embeddings_to_remove = [item for sublist in embeddings_to_remove for item in sublist]
            groups.append(found_words)
            sorted_keys = [k for k in sorted_keys if not any(i in embeddings_to_remove for i in k)]
        
        elif user_input == 'n':
            lives_left -= 1
            wrong_groups.append(found_words)
            sorted_keys = [k for k in sorted_keys if sum(embeddings_to_words[i] in connection for i in k) <= 2]
            if lives_left == 0:
                print()
                print("I found "+str(len(groups))+"/4 connections.")
                print("Better luck next time!")
                quit()
            print("Remaining tries: "+str(lives_left))
        
        elif user_input == '3':
            lives_left -= 1
            close_groups.append(found_words)
            sorted_keys = [k for k in sorted_keys if (sum(embeddings_to_words[i] in connection for i in k) in set([0,1,3]))]
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
