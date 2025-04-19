#import time
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.manifold import spectral_embedding
#from scipy.optimize import milp
#from scipy.optimize import Bounds
#from scipy.optimize import LinearConstraint
import numpy as np
#from sklearn.metrics.pairwise import linear_kernel
#from sklearn.metrics.pairwise import rbf_kernel
import csv
from itertools import combinations
import matplotlib.pyplot as plt
#import fasttext.util
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
#from sklearn.manifold import MDS
#from cluster_equal_size import cluster_equal_size_mincostmaxflow
#from freedictionaryapi.clients.sync_client import DictionaryApiClient
#from sentence_transformers.cross_encoder import CrossEncoder
#from sklearn.cluster import SpectralClustering
from scipy.linalg import qr

large_width = 400
np.set_printoptions(linewidth=large_width)

def get_definition(word):
    defns = []
    with DictionaryApiClient() as client:
        parser = client.fetch_parser(word)
    word = parser.word
    for meaning in word.meanings:
        for definition in meaning.definitions:
                defns.append(definition.definition)
    return(defns)

def read_csv(file_path):
    data_array = []
    try:
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data_array.append(row)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return []
    except Exception as e:
         print(f"An error occurred: {e}")
         return []
    return data_array

puzzle = read_csv("puzzles/test_puzzle_1.txt") #3 is easy, 2 is medium, 1,4 are hard
puzzle_words = [x.lower().strip() for xs in puzzle for x in xs]
print(np.reshape(puzzle_words,(4,4)))
print()

model = SentenceTransformer("all-MiniLM-L6-v2")
#model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
#model = CrossEncoder("cross-encoder/stsb-TinyBERT-L4")

#def find_max_position(matrix, idx):
#    matrix_np = np.array(matrix)
#    max_index_flat = np.argmax(matrix_np)
#    row_index, col_index = np.unravel_index(max_index_flat, matrix_np.shape)
#    return [int(row_index), int(col_index)][idx]

#def enhance_match(x,y):
#    if x > 0.95:
#        return 1.0
#    else:
#        return x*y

def compute_sim_array(embeddings):
    all_similarities = [[[] for j in embeddings] for i in embeddings]
    for xi,x in enumerate(embeddings):
        for yi,y in enumerate(embeddings):
            #doing extra work, but fast compared to embedding
            all_similarities[xi][yi] = cosine_similarity(x,y)
    return(all_similarities)

def score_defn(key, defn, words, weights, all_similarities):
    sim_measure = []
    for i,w in enumerate(words):
        word_sims = all_similarities[i]
#        word_sims = cosine_similarity([defn],w)[0]
#        weighted_sims = [enhance_match(x,y) for x,y in zip(word_sims, weights[i])]
        weighted_sims = [x*y for x,y in zip(word_sims, weights[i])]
        best_sim = max(weighted_sims)
        sim_measure.append(best_sim)
    sim_measure.sort()
    sim_measure.reverse()
    best_sims = sim_measure[0:3]
    worst_sims = sim_measure[3:]
#    if best_sims[0] > 0.95:
#        return 1.0
    return float(np.mean(best_sims) - np.mean(worst_sims))

def update_scores(defns, embeddings, scores, all_similarities):
    all_scores = []
    for wi,w in enumerate(embeddings):
        defn_scores = []
        words_to_check = [item for i, item in enumerate(embeddings) if i != wi]
        for wj,w_defn in enumerate(w):
            curr_similarities = [x[wj] for x in all_similarities[wi]]
            score = score_defn(defns[wi][wj], w_defn, words_to_check, [item for i, item in enumerate(scores) if i != wi], curr_similarities)
#            score = score_defn(w_defn, words_to_check)
            defn_scores.append(score)
        norm_score = [float(i)/sum(defn_scores) for i in defn_scores]
        updated_scores = [x*y for x,y in zip(norm_score,scores[wi])]
        updated_scores = [float(i)/sum(updated_scores) for i in updated_scores]
        all_scores.append(list(updated_scores))
    return all_scores

def get_best_defns(embeddings, definitions, all_similarities):
    all_scores = [[1.0/len(defn) for d in defn] for defn in definitions]
    for i in range(25): #ZMS
        all_scores = update_scores(definitions, embeddings, all_scores, all_similarities)
        best_defns = []
        for defn in all_scores:
            best_defns.append(defn.index(max(defn)))
    return best_defns

def compute_dist_mat(embeddings, definitions, all_similarities):
#    epsilon = 0.000001
    dist_mat2 = [[0.0 for j in range(len(embeddings))] for i in range(len(embeddings))]
    chosen_defns = get_best_defns(embeddings, definitions, all_similarities) 
    for wi,w in enumerate(embeddings):
        for xi,x in enumerate(embeddings):
            if xi <= wi:
                continue
            dist_mat2[wi][xi] = all_similarities[wi][xi][chosen_defns[wi]][chosen_defns[xi]]
            dist_mat2[xi][wi] = dist_mat2[wi][xi]
    for i in range(len(dist_mat2)):
        dist_mat2[i][i] = 1.0 #+ epsilon
    return dist_mat2








def flatten_off_diagonal(matrix):
    ret = []
    rows = len(matrix)
    cols = len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            if i < j:
                ret.append(matrix[i][j])
    return np.array(ret)

def make_fake_constraints(n_words): #taking convention that bl and bu are 0
    n_fake_vars = int((n_words**2 - n_words)/2)
    a_mat = np.array([[0.0 for j in range(n_fake_vars+n_words)] for i in range(3*n_fake_vars)])
    bu = np.array([0.0 for i in range(3*n_fake_vars)])
    bl = np.array([-1.0 for i in range(3*n_fake_vars)])
    i = 0
    fake_var_pos = n_words
    for v1 in range(n_words):
        for v2 in range(n_words):
            if v1 < v2: # there's some redundancies here -- don't need zij variables for definitions in the same word
                a_mat[i,v1] = -1
                a_mat[i,fake_var_pos] = 1
                i += 1
                a_mat[i,v2] = -1
                a_mat[i,fake_var_pos] = 1
                i += 1
                a_mat[i,fake_var_pos] = 1
                a_mat[i,v1] = -1
                a_mat[i,v2] = -1
#                bl[i] = -1
                i += 1
                fake_var_pos += 1
    return bu, bl, a_mat

#def find_closest_four(puzzle_words, my_dist_mat, wrong_groups, close_groups, word_to_working_word, working_word_to_word): 
#    start_time = time.perf_counter()
#    n_words = len(my_dist_mat)
#    P = np.array(my_dist_mat)
#    x_sum_rule = np.array([1.0 for i in range(n_words)])
#    c_vec = np.array([0.0 for i in range(n_words)])
#    x_fake_sum_rule = np.array([0.0 for i in range(int((n_words**2 - n_words)/2))])
#    x_sum_rule = np.append(x_sum_rule,x_fake_sum_rule) #first n_words are the variables we care about, rest are just for the optiization
#    c_vec_fake = flatten_off_diagonal(P)
#    c_vec = -2*np.append(c_vec,c_vec_fake)
#    bu, bl, a_mat = make_fake_constraints(n_words) #translate binary quadratic program to mixed integer linear program
#    a_mat = np.vstack([a_mat,x_sum_rule]) #restrict guesses to exactly 4 words
#    bl = np.append(bl,[4.])
#    bu = np.append(bu,[4.])
#    for g in wrong_groups: #resctrict guesses to not be too similar to previous wrong guesses
#        new_row = np.array([0.0 for i in a_mat[0]])
#        for w in g:
#            word_index = puzzle_words.index(w)
#            if isinstance(word_index,int):
#                new_row[word_index] = 1.0
#        a_mat = np.vstack([a_mat,new_row])
#        bl = np.append(bl,[0.])
#        bu = np.append(bu,[2.])
#    for g in close_groups: #restrict guesses using info about close guesses: guesses using close group members can have no more than 3 close group members
#        new_row = np.array([0.0 for i in a_mat[0]])
#        for w in g:
#            word_index = puzzle_words.index(w)
#            if isinstance(word_index,int):            
#                new_row[word_index] = 1.0
#        a_mat = np.vstack([a_mat,new_row])
#        bl = np.append(bl,[3.])
#        bu = np.append(bu,[3.])
##    for g in close_groups: #restrict guesses using info about close guesses: new groups should have at most one member of close group -- only way to include is nonlinear and this doesn't happen much in practice
##        new_row = np.array([0.0 for i in a_mat[0]])
##        not_g = [i for i in range(n_words) if i not in g]
##        for w in not_g:
##            word_index = puzzle_words.index(w)
##            if isinstance(word_index,int):            
##                new_row[word_index] = 1.0
##        a_mat = np.vstack([a_mat,new_row])
##        bl = np.append(bl,[3.])
##        bu = np.append(bu,[4.])
#    for restriction in word_to_working_word:
#        new_row = np.array([0.0 for i in a_mat[0]])
#        new_row[restriction] = 1.0
#        a_mat = np.vstack([a_mat,new_row])
#        bl = np.append(bl,[0.])
#        bu = np.append(bu,[1.])
#    l = np.array([0.0 for i in range(len(c_vec))])
#    u = np.array([1.0 for i in range(len(c_vec))])
#    need_int = np.array([1.0 for i in range(len(c_vec))])
#    end_time = time.perf_counter()
#    print("MILP Setup time")
#    print(end_time-start_time)
#    start_time = time.perf_counter()
#    solution = milp(c_vec, integrality=need_int, bounds=Bounds(lb=l,ub=u), constraints=LinearConstraint(a_mat, lb=bl, ub=bu))#, options={"mip_rel_gap":5})
#    end_time = time.perf_counter()
#    print("MILP time:")
#    print(end_time-start_time)
#    print(solution)
##    print(solution.x)
##    test = x_sum_rule.copy()
##    test[[4,5,6,7,8,9,10,11,12,13,14,15]] = 0.0
##    test[[16,17,18,31,32,45]] = 1.0
##    print(np.dot(c_vec,test))
##    test[[2,3,6,7,8,9,10,11,12,13,14,15]] = 0.0
##    test[[16,19,20,33,34,45+25]] = 1.0 
##    print(np.dot(c_vec,test))
#    x = [i for i,a in enumerate(solution.x) if a > 0.99 and i < n_words]
#    print(x)
#    actual_words = [working_word_to_word[i] for i in x]
#    print(actual_words)
##    print(np.dot(np.dot(np.array(solution.x[0:16]).T,P),np.array(solution.x[0:16])))
##    print(np.dot(np.dot(np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]).T,P),np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])))
#    return x, solution.fun 

#def permute_matrix(matrix, permutation):
#    permuted_matrix = matrix[permutation, :]
#    permuted_matrix = permuted_matrix[:, permutation]
#    return permuted_matrix
#
#def search_connections(puzzle_words, my_dist_mat, wrong_groups, close_groups, word_to_working_word, working_word_to_word):
#    word_shuffles = []
#    permutations = []
#    for i in range(1):
#        shuffle_list = [int(i) for i in range(len(puzzle_words))]
#        random.shuffle(shuffle_list)
#        permutations.append(list(shuffle_list))
#        word_shuffles.append([puzzle_words[i] for i in shuffle_list])
#    solutions = []
#    costs = []
#    for p,l in zip(permutations,word_shuffles):
#        soln,cost = find_closest_four(l, permute_matrix(np.array(my_dist_mat),p), wrong_groups, close_groups, word_to_working_word, working_word_to_word)
#        solutions.append(soln)
#        costs.append(cost)
#    min_index, min_value = min((idx, val) for (idx, val) in enumerate(costs))
#    final_guess = solutions[min_index]
#    final_idx = []
#    for g in final_guess:
#        guess_word = word_shuffles[min_index][g]
#        word_index = puzzle_words.index(guess_word)
#        final_idx.append(word_index)
#    return final_idx

def make_sim_mat(embeddings, definitions):
    sim_mat = cosine_similarity(embeddings)
    low_idx = 0
    high_idx = 0
    for i,d in enumerate(definitions):
        new_diag = np.eye(len(d)) - np.ones((len(d),len(d)))
        high_idx += len(d) 
        sim_mat[low_idx:high_idx,low_idx:high_idx] = new_diag
        low_idx = high_idx
    return sim_mat

#def find_closest_four(working_words, sim_mat, wrong_groups, close_groups, word_to_working_word, working_word_to_word): #add constraints
#    n_points = len(sim_mat)
#    best_sim = 0
#    best_set = []
#    for i in combinations(range(n_points),4):
#        curr_sim = np.sum(sim_mat[np.ix_(i,i)])
#        if curr_sim > best_sim:
#            best_sim = curr_sim
#            best_set = i
#    return best_set

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

#            sorted_keys = [k for k in sorted_keys if sum(i in opt_solution for i in k) <= 2] #wrong groups
#            sorted_keys = [k for k in sorted_keys if sum(i in opt_solution for i in k) % 2 == 1] #close groups

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



def select_clusters(words, model, n_defns):
    synsets = [wn.synsets(w) for w in words]
    all_definitions = [[w.definition() for w in s] for i,s in enumerate(synsets)]
    embeddings = []
    definitions = []
    for i,d in enumerate(all_definitions):
        if len(d) < n_defns:
            d.append(words[i])
        encodings = model.encode(d)
        _, _, qr_selection = qr(np.array(encodings).T, pivoting=True)
        qr_selection = qr_selection[0:n_defns]
        embeddings.append(encodings[qr_selection])
        chosen_defns = [d[j] for j in qr_selection]
        print(chosen_defns)
        definitions.append(chosen_defns)
#    n_defn = [len(definitions[i]) for i,w in enumerate(words)]
    working_words = [item for sublist in definitions for item in sublist]
    working_word_to_word = [i for i,w in enumerate(words) for j in definitions[i]]
    word_to_working_word = [[] for i in range(len(words))]
    counter = 0
    for i,w in enumerate(words):
        for d in definitions[i]:
            word_to_working_word[i].append(counter)
            counter += 1
#    print(word_to_working_word)
    embeddings = [item for sublist in embeddings for item in sublist] #model.encode(working_words)
    sim_mat = make_sim_mat(embeddings, definitions)
#    for i,row in enumerate(sim_mat):
#        idx = len(sim_mat) - i
#        row_max = max(row)
#        if row_max < 0.3 and len(working_words[idx]) > 15:
#            sim_mat = np.delete(sim_mat, idx, axis=0)
#            sim_mat = np.delete(sim_mat, idx, axis=1)

#    plt.matshow(sim_mat)
#    plt.colorbar()
#    plt.title("All Similarities")
#    plt.show()
#    for row in sim_mat:
#        print([float(np.mean(row)),float(max(row))])
#    return "done"
    all_groups = enumerate_groups(sim_mat)
    sorted_keys = sorted(all_groups, key=all_groups.get, reverse=True)
    groups = []
    wrong_groups = []
    close_groups = []
    lives_left = 4
    while lives_left > 0 and len(groups) < 4:
#        sim_mat = make_sim_mat(embeddings, definitions) #want to remove at some point
#        plt.matshow(sim_mat)
#        plt.colorbar()
#        plt.title("All Similarities")
#        plt.show()
#        opt_solution = search_connections(working_words, sim_mat, wrong_groups, close_groups, word_to_working_word, working_word_to_word)
#        opt_solution, solution_cost = find_closest_four(working_words, sim_mat, wrong_groups, close_groups, word_to_working_word, working_word_to_word)
#        opt_solution = find_closest_four(working_words, sim_mat, wrong_groups, close_groups, word_to_working_word, working_word_to_word)
        if sorted_keys == []:
            raise Exception("I ran out of options!")
        opt_solution = list(sorted_keys[0])
#        opt_solution = find_cluster(sim_mat, 4, wrong_groups, close_groups, working_word_to_word, words)
#        print(opt_solution)
        cl1 = [working_word_to_word[i] for i in opt_solution]
#        print(cl1)
        found_words = [w for i,w in enumerate(words) if i in cl1]
        print(found_words)
        if len(found_words) != 4:
            raise Exception("I messed up somewhere!")
        #filter out possibilities from sorted kerys here without changing order, no need to update anything
        while True:
            user_input = input("Is this a valid connection? (y/n/3)")
            if user_input == 'y' or user_input == 'n' or user_input == '3':
                break
            else:
                print("Please enter a valid input. (y/n/3)")
                continue
        if user_input == 'y': #something not updated correctly here?
            defns_to_remove = [word_to_working_word[i] for i in cl1]
            defns_to_remove = [item for sublist in defns_to_remove for item in sublist]
            groups.append(found_words)
            sorted_keys = [k for k in sorted_keys if not any(i in defns_to_remove for i in k)]
#            print(len(sorted_keys))
#            words = [w for i,w in enumerate(words) if i not in cl1]
#            working_words = [w for i,w in enumerate(working_words) if i not in defns_to_remove]
##            working_word_to_word = [w for i,w in enumerate(working_word_to_word) if i not in defns_to_remove]
##            word_to_working_word = [w for i,w in enumerate(word_to_working_word) if i not in cl1]
#            embeddings = [e for i,e in enumerate(embeddings) if i not in defns_to_remove]
#            definitions = [d for i,d in enumerate(definitions) if i not in cl1]
#            working_word_to_word = [i for i,w in enumerate(words) for j in definitions[i]]
#            word_to_working_word = [[] for i in range(len(words))]
#            counter = 0
#            for i,w in enumerate(words):
#                for d in definitions[i]:
#                    word_to_working_word[i].append(counter)
#                    counter += 1
#            sim_mat = np.array([[col for j,col in enumerate(row) if j not in defns_to_remove] for i,row in enumerate(sim_mat) if i not in defns_to_remove])
#            all_similarities = [[k for i,k in enumerate(s) if i not in cl1] for j,s in enumerate(all_similarities) if j not in cl1]
        elif user_input == 'n':
            lives_left -= 1
            wrong_groups.append(found_words)
            sorted_keys = [k for k in sorted_keys if sum(words[working_word_to_word[i]] in found_words for i in k) <= 2]
            if lives_left == 0:
                print("Better luck next time!")
                quit()
            print("Remaining tries: "+str(lives_left))
        elif user_input == '3':
            lives_left -= 1
            close_groups.append(found_words)
            sorted_keys = [k for k in sorted_keys if (sum(words[working_word_to_word[i]] in found_words for i in k) in set([0,1,3]))]
            if lives_left == 0:
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
best_guess = np.array(select_clusters(test_words, model, 5))
