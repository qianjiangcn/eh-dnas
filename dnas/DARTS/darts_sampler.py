import pickle
import random
import sys
import os
sys.path.insert(0, os.getcwd())


def sample_one_vec(out_type):
    non_zero_edges = []
    ops_of_node = []
    for ed in ([0, 1], [2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13]):
        non_zero_edges.extend(random.sample(ed, 2))
        ops_of_node.extend(random.sample([*range(1, 7)], 2))
    vec = [int(0)]*14
    for edge, op in zip(non_zero_edges, ops_of_node):
        vec[edge] = int(op)
    if out_type == 'int':
        vec = int(''.join([str(i) for i in vec]))
    elif out_type == 'str':
        vec = ''.join([str(i) for i in vec])
    return vec


def check_unique_rate(my_list):
    seen = set()
    uniq = []
    for x in my_list:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    print('Uniq rate is {}'.format(len(uniq)/len(my_list)))
    return uniq


def sample_input_vec(num, seen=None):
    my_dict = {}
    uniq_list = []
    if seen is None:
        seen = []
        print('Initialize seen as empty list')
    for i in range(num):
        geno_vec = sample_one_vec(out_type='int')
        while geno_vec in seen:
            geno_vec = sample_one_vec(out_type='int')
        seen.append(geno_vec)
        my_dict[str(i)] = geno_vec
        uniq_list.append(geno_vec)
    return my_dict, uniq_list, seen


def save_dict_to_pickle(dic, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(dic, f)


def sample_vecs_checklater():
    my_dict = {}
    my_list = []
    num = 1420000
    for i in range(num):
        geno_vec1 = sample_one_vec(out_type='str')
        geno_vec2 = sample_one_vec(out_type='str')
        geno_vec = geno_vec1+geno_vec2
        my_dict[str(i)] = geno_vec
        my_list.append(int(geno_vec))
    uniq = check_unique_rate(my_list)
    print(len(uniq))
    train_list = uniq[:1000000]
    val_list = uniq[1000000:1200000]
    test_list = uniq[1200000:1400000]
    dict_train = {str(i): train_list[i] for i in range(len(train_list))}
    dict_val = {str(i): val_list[i] for i in range(len(val_list))}
    dict_test = {str(i): test_list[i] for i in range(len(test_list))}

    pk_path_train = 'dnas/DARTS/hwdataset_100w/100w_input_train.pkl'
    pk_path_val = 'dnas/DARTS/hwdataset_100w/100w_input_val.pkl'
    pk_path_test = 'dnas/DARTS/hwdataset_100w/100w_input_test.pkl'
    save_dict_to_pickle(dict_train, pk_path_train)
    save_dict_to_pickle(dict_val, pk_path_val)
    save_dict_to_pickle(dict_test, pk_path_test)

sample_vecs_checklater()
