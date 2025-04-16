from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import spectral_embedding
import numpy as np
#from sklearn.metrics.pairwise import linear_kernel
#from sklearn.metrics.pairwise import rbf_kernel
import csv
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
#import fasttext.util
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.manifold import MDS
from cluster_equal_size import cluster_equal_size_mincostmaxflow
#from freedictionaryapi.clients.sync_client import DictionaryApiClient
#from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.cluster import SpectralClustering

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

puzzle = read_csv("puzzles/test_puzzle_3.txt") #2,3 are easy, 1,4 are hard
puzzle_words = [x.lower().strip() for xs in puzzle for x in xs]
print(np.reshape(puzzle_words,(4,4)))
print()

#fasttext.util.download_model('en', if_exists='ignore')  
#ft = fasttext.load_model('cc.en.300.bin')
#
#vecs = [ft.get_word_vector(w) for w in puzzle_words]

#sim_mat = cosine_similarity(vecs)
#sim_mat = linear_kernel(vecs)
#sim_mat = rbf_kernel(vecs, gamma=1/10)
model = SentenceTransformer("all-MiniLM-L6-v2")
#model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
#model = CrossEncoder("cross-encoder/stsb-TinyBERT-L4")

#synsets = [wn.synsets(w) for w  in puzzle_words]
##definitions = [get_definition(puzzle_words[i]) + [w.definition() for w in s] for i,s in enumerate(synsets)]
#definitions = [[w.definition() for w in s] for i,s in enumerate(synsets)]
##definitions = [[] for i,s in enumerate(synsets)]
##definitions = [get_definition(w) for w in puzzle_words]
#for i,d in enumerate(definitions):
#    d.append(puzzle_words[i])
#
#dist_mat2 = [[0.0 for j in range(16)] for i in range(16)]
#embeddings = [model.encode(defn) for defn in definitions]

#print()
#for d in definitions:
#    print(d)

def find_max_position(matrix, idx):
    matrix_np = np.array(matrix)
    max_index_flat = np.argmax(matrix_np)
    row_index, col_index = np.unravel_index(max_index_flat, matrix_np.shape)
    return [int(row_index), int(col_index)][idx]

#def enhance_match(x,y):
#    if x > 0.95:
#        return 1.0
#    else:
#        return x*y

def score_defn(key, defn, words, weights, sim_dic):
    sim_measure = []
    for i,w in enumerate(words):
#        key = 1
        if key not in sim_dic:
            sim_dic[key+str(i)] = cosine_similarity([defn],w)[0]
        word_sims = sim_dic[key+str(i)]
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

def update_scores(defns, embeddings, scores, sim_dic):
    all_scores = []
    for wi,w in enumerate(embeddings):
        defn_scores = []
        words_to_check = [item for i, item in enumerate(embeddings) if i != wi]
        for wj,w_defn in enumerate(w):
            score = score_defn(defns[wi][wj], w_defn, words_to_check, [item for i, item in enumerate(scores) if i != wi], sim_dic)
#            score = score_defn(w_defn, words_to_check)
            defn_scores.append(score)
        norm_score = [float(i)/sum(defn_scores) for i in defn_scores]
        updated_scores = [x*y for x,y in zip(norm_score,scores[wi])]
        updated_scores = [float(i)/sum(updated_scores) for i in updated_scores]
        all_scores.append(list(updated_scores))
    return all_scores

def get_best_defns(embeddings, definitions, sim_dic):
    all_scores = [[1.0/len(defn) for d in defn] for defn in definitions]
    for i in range(25):
        all_scores = update_scores(definitions, embeddings, all_scores, sim_dic)
        best_defns = []
        for defn in all_scores:
            best_defns.append(defn.index(max(defn)))
    return best_defns

def compute_dist_mat(embeddings, definitions, sim_dic):
    dist_mat2 = [[0.0 for j in range(len(embeddings))] for i in range(len(embeddings))]
    chosen_defns = get_best_defns(embeddings, definitions, sim_dic) 
    for wi,w in enumerate(embeddings):
        for xi,x in enumerate(embeddings):
            if xi <= wi:
                continue
            dist_mat2[wi][xi] = np.abs(cosine_similarity([w[chosen_defns[wi]]],[x[chosen_defns[xi]]])[0][0])
            dist_mat2[xi][wi] = dist_mat2[wi][xi]
    for i in range(len(dist_mat2)):
        dist_mat2[i][i] = 1.0
    return dist_mat2

#dist_mat = compute_dist_mat(embeddings, definitions)

def find_closest_four(my_dist_mat):
    curr_best_dist = -10
    best_idx_a = 0
    for i in range(len(my_dist_mat)):
        curr_row = my_dist_mat[i].copy()
        curr_row.sort()
        curr_row.reverse()
        curr_score = np.mean(curr_row[0:4])
        if curr_score > curr_best_dist:
            curr_best_dist = curr_score
            best_idx_a = i
        i += 1
    return(np.argsort(my_dist_mat[best_idx_a])[-4:])

#def select_clusters(words, dist_mat):
#    dist_mat2 = dist_mat.copy()
#    groups = []
#    for i in range(4):
#        cl1 = find_closest_four(dist_mat2)
#        dist_mat2 = [[x for i,x in enumerate(l) if i not in cl1] for li,l in enumerate(dist_mat2) if li not in cl1] #should completely recompute dist_mat here
#        groups.append([w for i,w in enumerate(words) if i in cl1])
#        words = [w for i,w in enumerate(words) if i not in cl1]
#    return(groups)

def select_clusters(words, model):
    synsets = [wn.synsets(w) for w in words]
    definitions = [[w.definition() for w in s] for s in synsets]
    for i,d in enumerate(definitions):
        d.append(words[i])
    embeddings = [model.encode(defn) for defn in definitions]
    sim_dic = {}
    groups = []
    for i in range(3):
        dist_mat2 = compute_dist_mat(embeddings, definitions, sim_dic)
        plt.matshow(dist_mat2)
        plt.colorbar()
        plt.title("Word Similarities defn Transformer")
        plt.show()
        cl1 = find_closest_four(dist_mat2)
#        dist_mat2 = [[x for i,x in enumerate(l) if i not in cl1] for li,l in enumerate(dist_mat2) if li not in cl1] #should completely recompute dist_mat here
        found_words = [w for i,w in enumerate(words) if i in cl1]
        groups.append(found_words)
        print(found_words)
        words = [w for i,w in enumerate(words) if i not in cl1]
        embeddings = [e for i,e in enumerate(embeddings) if i not in cl1]
        definitions = [d for i,d in enumerate(definitions) if i not in cl1]
        dist_mat2.clear()
    print(words)
    groups.append(words)
    return(groups)

best_guess = np.array(select_clusters(puzzle_words, model))

#all_scores = [[1.0/len(defn) for d in defn] for defn in definitions]
#for i in range(25):
#    all_scores = update_scores(definitions, embeddings, all_scores)
#    best_defns = []
#    for defn in all_scores:
#        best_defns.append(defn.index(max(defn)))
##    print(best_defns)
#    print(['{0:.2f}'.format(max(a)) for a in all_scores])
#
#print()
#
#print(best_defns)
#chosen_defns = [0,1,0,2,2,2,1,4,3,7,5,2,4,2,0,1] # chosen by hand to maximize overlap with group and minimze overlap with other groups
#print(chosen_defns)
#chosen_defns = best_defns
#print(['{0:.2f}'.format(max(a)) for a in all_scores])
#
#for a in all_scores:
#    print(['{0:.2f}'.format(n) for n in a])
#print()
#
#save_pair = [5,6]
#save_sims = []
#defns_used = [['x' for j in range(16)] for i in range(16)]
##defns_used2 = [[0 for j in range(16)] for i in range(16)]
##chosen_defns = [0,1,0,2,2,2,1,4,3,7,5,2,4,2,0,1] # chosen by hand to maximize overlap with group and minimze overlap with other groups
#for wi,w in enumerate(embeddings):
#    for xi,x in enumerate(embeddings):
#        if xi <= wi:
#            continue
#        all_sims = [[0.0 for j in x] for i in w]
#        for wj,ws in enumerate(w):
#            for xj,xs in enumerate(x):
#                all_sims[wj][xj] = np.abs(cosine_similarity([ws],[xs])[0][0])
##                all_sims.append(cosine_similarity([ws],[xs])[0][0])
##        dist_mat2[wi][xi] = 1-max(all_sims) #uncomment to make dist matrix
##        print([wi,xi])
##        print(all_sims)
#        if wi == save_pair[0] and xi == save_pair[1]:
#            save_sims = all_sims
##        dist_mat2[wi][xi] = max([max(x) for x in all_sims]) #uncomment to make affinity matrix
#        dist_mat2[wi][xi] = all_sims[chosen_defns[wi]][chosen_defns[xi]] #uncomment to make affinity matrix
#        defns_used[wi][xi] = find_max_position(all_sims,0) 
#        defns_used[xi][wi] = find_max_position(all_sims,1)
##        dist_mat2[wi][xi] = max(all_sims) #uncomment to make affinity matrix
##        dist_mat2[wi][xi] = np.mean(all_sims) #uncomment to make affinity matrix
##        dist_mat2[wi][xi] = np.mean([x**2 for x in all_sims]) #uncomment to make affinity matrix
#        dist_mat2[xi][wi] = dist_mat2[wi][xi]
#
#print("looking for consistency in number accross rows within (expected) cluster")
#print(np.array(defns_used))

#print(save_sims)
#plt.hist(sum(save_sims,[]))
#plt.show()

#for wi,w in enumerate(definitions):
#    for xi,x in enumerate(definitions):
#        if xi < wi:
#            continue
#        all_sims = []
#        for ws in w:
#            ranks = model.rank(ws, x)
#            scores = [rank['score'] for rank in ranks]
#            all_sims.append(max(scores))
##        for xs in x:
##            ranks = model.rank(xs, w)
##            scores = [rank['score'] for rank in ranks]
##            all_sims.append(max(scores))
##        all_sims = max(all_sims)
#        dist_mat2[wi][xi] = 1-max(all_sims)
#        dist_mat2[xi][wi] = dist_mat2[wi][xi]

#for i,row in enumerate(dist_mat2):
#    dist_mat2[i][i] = 1.0

#cl1 = find_closest_four(dist_mat2)
#print(cl1)
#new_dist_mat = [[0.0 for i in range(16)] for j in range(16)]
#for i,r in enumerate(new_dist_mat):
#    for j,c in enumerate(r):
#        if i not in cl1 and j not in cl1:
#            new_dist_mat[i][j] = dist_mat2[i][j]
#cl2 = find_closest_four(new_dist_mat)
#print(cl2)
#new_dist_mat2 = [[0.0 for i in range(16)] for j in range(16)]
#for i,r in enumerate(new_dist_mat2):
#    for j,c in enumerate(r):
#        if i not in cl2 and j not in cl2:
#            new_dist_mat2[i][j] = new_dist_mat[i][j]
#cl3 = find_closest_four(new_dist_mat2)
#print(cl3)
#new_dist_mat3 = [[0.0 for i in range(16)] for j in range(16)]
#for i,r in enumerate(new_dist_mat3):
#    for j,c in enumerate(r):
#        if i not in cl3 and j not in cl3:
#            new_dist_mat3[i][j] = new_dist_mat2[i][j]
#cl4 = find_closest_four(new_dist_mat3)
#print(cl4)

#cluster_size = 4
#n_clusters = 4
#unassigned = range(cluster_size*n_clusters)
#clusters = [[] for n in n_clusters]
#for 

#mds = MDS(n_components=100, dissimilarity='precomputed', random_state=0)
# Get the embeddings
#X_transform = mds.fit_transform(dist_mat2)

#eigenvecs = spectral_embedding(np.array(dist_mat2), n_components=4, random_state=0)
#print(eigenvecs)

#clustering = SpectralClustering(n_clusters=4,
#        assign_labels='discretize',
#        affinity='precomputed',
#        random_state=0).fit(dist_mat2)
#predicted_groups = clustering.labels_

#clustering = SpectralClustering(n_clusters=3,
#        assign_labels='discretize',
#        affinity='precomputed',
#        random_state=0).fit([x[0:12] for x in dist_mat2[0:12]])
#predicted_groups = clustering.labels_
#predicted_groups = cluster_equal_size_mincostmaxflow(X_transform, 4, show_plt=False)[0]#[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
#predicted_groups = cluster_equal_size_mincostmaxflow(eigenvecs, 4, show_plt=False)[0]#[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
#print(predicted_groups)
#
#def group_by_cluster(points, labels):
#    grouped_points = defaultdict(list)
#    for i, label in enumerate(labels):
#        grouped_points[label].append(points[i])
#    return dict(grouped_points)
#
#def average_distance_in_group(group):
#    if len(group) < 2:
#        return 0
#    distances = [np.sqrt(sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))
#                 for p1, p2 in combinations(group, 2)]
#    return float(np.mean(distances))
#
#grouped_data = group_by_cluster(eigenvecs, predicted_groups)
#grouped_words = group_by_cluster(puzzle_words, predicted_groups)
#print(grouped_words)
#spreads = []
#for g in list(grouped_data.values()):
#    spreads.append(average_distance_in_group(g))
#print(spreads)

#smallest_cluster = spreads.index(min(spreads))

##plt.matshow(sim_mat)
##plt.colorbar()
##plt.title("Word Similarities synsets")
##plt.show()
#sizes = [64 for i in range(16)]
#colors = ['y','y','y','y','g','g','g','g','b','b','b','b','m','m','m','m']
#group_colors = [['y','g','b','m'][i] for i in predicted_groups]
#fig = plt.figure()
##ax = fig.add_subplot(projection='3d')
#ax = fig.add_subplot()
##ax.scatter(X_transform[:,0], X_transform[:,1], X_transform[:,2], s=sizes, c=colors)
#ax.scatter(X_transform[:,0], X_transform[:,1], s=[4*x for x in sizes], c=group_colors)
#ax.scatter(X_transform[:,0], X_transform[:,1], s=[2*x for x in sizes], c=['k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k'])
#ax.scatter(X_transform[:,0], X_transform[:,1], s=sizes, c=colors)
#plt.title('Embedding')
#plt.show()

#plt.matshow(dist_mat2)
#plt.colorbar()
#plt.title("Word Similarities defn Transformer")
#plt.show()


