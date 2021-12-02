import numpy as np
import pandas as pd
import networkx as nx

def read_data(edgeFile, typeFile, typeName):
    G = nx.from_edgelist(pd.read_csv(edgeFile, encoding='utf-8', header=None, sep='\t').values.tolist())
    nodeTypeList = []
    idList = np.loadtxt(typeFile, dtype=np.int)
    for i in idList[:,0]:
        for j in range(idList[i,1], idList[i,2]+1):
            nodeTypeList.append(typeName[i])
    return G, nodeTypeList

def save_embeddings(embed, NTL, embFile):
    idList = [NTL[i]+str(i) for i in range(embed.shape[0])]
    with open(embFile,'w',encoding='utf-8') as ef:
        ef.write('{} {}\n'.format(embed.shape[0], embed.shape[1]))
        for i in range(embed.shape[0]):
            ef.write('{} {}\n'.format(idList[i], ' '.join([str(j) for j in embed[i]])))

def partition(targetList, NumParts):
    partSize = len(targetList) / float(NumParts)
    return [ targetList[int(round(partSize * i)): int(round(partSize * (i + 1)))] for i in range(NumParts) ]