import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd()+'/model_zoo/DARTS')
import genotypes
from genotypes import *
from model_zoo.DARTS.model_search import Network
from model_zoo.DARTS.model import NetworkCIFAR, NetworkImageNet



def darts(arch, geno, cifar=True):
    steps = 4
    multiplier = 4
    if arch is not None:
        vec = [int(i) for i in str(arch)]
        geno_normal = parse_from_vec(vec[:14])
        geno_reduce = parse_from_vec(vec[14:])
        concat = range(2+steps-multiplier, steps+2)
        genotype = Genotype(
            normal=geno_normal, normal_concat=concat,
            reduce=geno_reduce, reduce_concat=concat
        )
    if geno is not None:
        genotype = eval("genotypes.%s" % geno)
    if cifar:
        layers = 8
        init_channels = 16
        CIFAR_CLASSES = 10
        auxiliary = True
        model = NetworkCIFAR(
            init_channels, CIFAR_CLASSES, layers, auxiliary, genotype)
    else:
        layers = 14
        init_channels = 48
        CIFAR_CLASSES = 1000
        auxiliary = True
        model = NetworkImageNet(
            init_channels, CIFAR_CLASSES, layers, auxiliary, genotype)

    return model


def parse(weights):
    gene = []
    n = 2
    start = 0
    steps = 4
    for i in range(steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -
                       max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
            k_best = None
            for k in range(len(W[j])):
                if k != PRIMITIVES.index('none'):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
            gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
    return gene


def parse_from_vec(vec):
    gene = []
    edge_input = [0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]
    for inp, op in zip(edge_input, vec):
        if op != PRIMITIVES.index('none'):
            gene.append((PRIMITIVES[op], inp))
    return gene
