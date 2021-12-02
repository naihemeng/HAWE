import networkx as nx
import random, time, math, os, sys
import numpy as np
import pandas as pd
import re
from collections import Counter
from itertools import product
from tqdm import tqdm, trange
from utils import read_data, partition
import torch
from torch.optim import Adam
from concurrent.futures import ProcessPoolExecutor, as_completed
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
import math

class HeteroGraph(object):
    '''
    Requestion: The node id of the input graph should start with 0. 
    '''
    def __init__(self, edgeFile, typeFile, typeName):

        self.APAW = dict() # all possible anonymous walks
        self.APHAW = dict() # all possible heterogeneous anonymous walks

        self.HG, self.NTL = read_data(edgeFile, typeFile, typeName)
        self.numNodes = len(self.HG.nodes())
        self.numEdges = len(self.HG.edges())
        print("The Heterogeneous graph has {} nodes & {} edges.".format(self.numNodes, self.numEdges))

    '''def read_data(self, edgeFile, typeFile, typeName):
        G = nx.from_edgelist(pd.read_csv(edgeFile, encoding='utf-8', header=None, sep='\t').values.tolist())
        nodeTypeList = []
        idList = np.loadtxt(typeFile, dtype=np.int)
        for i in idList[:,0]:
            for j in range(idList[i,1], idList[i,2]+1):
                nodeTypeList.append(typeName[i])
        return G, nodeTypeList'''

    def generate_subgraphs(self, hop = 2):
        '''For each node, generate a subgraph induced by nodes no more than 2 hops away from it.'''
        self.subGs = []
        for n in range(self.numNodes):
            nodesInSubG = set([n])
            addNodesLastTime = nodesInSubG
            for _ in range(1,hop+1):
                tmpNodes = []
                for i in addNodesLastTime:
                    tmpNodes += self.HG[i]
                addNodesLastTime = set(tmpNodes)
                nodesInSubG = nodesInSubG | addNodesLastTime
            self.subGs.append(self.HG.subgraph(nodesInSubG))
        print('The {}-hop node-centric subgraphs for all the nodes have been generated.'.format(hop))

    def schema_graph(self):
        relationSet = set([])
        for n1, n2 in self.HG.edges():
            relationSet = relationSet | set([ (self.NTL[n1], self.NTL[n2]) ])
        self.schG = nx.from_edgelist(list(relationSet))
        return self.schG

    def ap_AW(self, length):
        '''Get all possible anonymous walks of given length.'''
        APWalks = []
        lastStepWalks = [[0, 1]]
        for i in range(2, length+1):
            currentStepWalks = []
            for j in range(i + 1):
                for tmpWalks in lastStepWalks:
                    if tmpWalks[-1] != j and j <= max(tmpWalks) + 1:
                        APWalks.append(tmpWalks + [j])
                        currentStepWalks.append(tmpWalks + [j])
            lastStepWalks = currentStepWalks
        APWalks = list(filter(lambda walk: len(walk) ==  length + 1, APWalks))
        self.APAW[length] = APWalks
        print('APAW of length {} has been obtained. The total number is {}.'.format(length, len(self.APAW[length]) ) )

    def ap_HAW(self, length):
        '''Get all possible heterogeneous anonymous walks of given length. '''
        if length not in self.APAW or len(self.APAW[length]) <= 0:
            self.ap_AW(length)
        types = list(product(set(self.NTL), repeat=length+1))
        self.APHAW[length] = [list(zip(item[0],item[1]))  for item in product(types,self.APAW[length])]
        print('APHAW of length {} has been obtained. The total number is {}.'.format(length, len(self.APHAW[length]) ) )

    #def random_walk(self, subG, node, length):

    def create_random_walk_graph(self, subG):
        '''Create a probabilistic graph for the given graph.'''
        RwG = nx.DiGraph()
        for node in subG:
            edges = subG[node]
            total = float(sum([edges[n].get('weight', 1) for n in edges if n != node]))
            for n in edges:
                if n != node:
                    RwG.add_edge(node, n, weight= edges[n].get('weight', 1)/total)
        return RwG

    def next_step_node(self, RwG, node):
        '''Moves one step from the current node.'''
        
        r = random.uniform(0, 1)
        low = 0
        for n in RwG[node]:
            p = RwG[node][n]['weight']
            if r <= low + p:
                return n
            low += p

    def hetero_anonymous_walk(self, RwG, node, length):
        '''Creates heterogeneuos anonymous walk of the given length from a node in a graph.'''
        d = dict()
        d[node] = 0
        count = 1
        walk = [(self.NTL[node], d[node])]
        for i in range(length):
            n = self.next_step_node(RwG, node)
            if n not in d:
                d[n] = count
                count += 1
            walk.append((self.NTL[n], d[n]))
            node = n
        return tuple(walk)

    def coarse_hetero_anonymous_walk(self, RwG, node, length):
        '''Creates heterogeneuos anonymous walk of the given length from a node in a graph.'''
        d = dict()
        d[node] = 0
        count = 1
        walk = [d[node]]
        c = {'source':self.NTL[node]}
        for i in range(length):
            n = self.next_step_node(RwG, node)
            if self.NTL[n] not in c:
                c[self.NTL[n]] = 0
            if n not in d:
                d[n] = count
                count += 1
                c[self.NTL[n]] += 1
            walk.append(d[n])
                
            node = n
        for k in c:
            walk.append(str(k)+'-'+str(c[k]))
        return tuple(walk)

    def _anonymous_walk(self, RwG, node, length):
        '''Creates anonymous walk of the given length from a node in a graph.'''
        d = dict()
        d[node] = 0
        count = 1
        walk = [d[node]]
        for i in range(length):
            n = self.next_step_node(RwG, node)
            if n not in d:
                d[n] = count
                count += 1
            walk.append(d[n])
            node = n
        return tuple(walk)

    def walk2pattern(self, walk):
        '''Converts a walk to heterogeneous anonymous walks.'''
        idx = 0
        pattern = []
        d = dict()
        for node in walk:
            if node not in d:
                d[node] = idx
                idx += 1
            pattern.append((self.NTL[node], d[node]))
        return tuple(pattern)    

    def feature_based_subgraph_embedding_nonsampling(self, subG, length):
        '''Computing the feature-based HAWE for the given subgraph without sampling.'''
        givenRwG = self.create_random_walk_graph(subG)
        walks = dict()
        occurredWalks = []

        def patterns(RwG, node, length, walks, current_walk=None, current_dist=1):
            if current_walk is None:
                current_walk = [node]
            if len(current_walk) > 1:
                occurredWalks.append(current_walk)
                currentPattern = self.walk2pattern(current_walk)
                amount = current_dist
                amount /= len(RwG)
                walks[currentPattern] = walks.get(currentPattern, 0) + amount
            if length > 0:
                for n in RwG[node]:
                    patterns(RwG, n, length-1, walks, current_walk+[n], current_dist*RwG[node][n]['weight'])
        
        for node in givenRwG.nodes():
            patterns(givenRwG, node, length, walks)
        return walks

    def feature_based_subgraph_embedding_sampling(self, subG, length, numSamples):
        givenRwG = self.create_random_walk_graph(subG)
        walks = dict()
        amount = 1./ numSamples
        for it in range(numSamples):
            node = np.random.choice(givenRwG.nodes())
            haw = self.hetero_anonymous_walk(givenRwG, node, length)
            #if it > 4:
            #    break
            for l in range(3, len(haw) + 1):
                w_cropped = haw[:l]
                if w_cropped not in walks:
                    walks[w_cropped] = amount
                else:
                    walks[w_cropped] += amount
        return walks


    def feature_based_subgraph_embedding(self, length = 5, hop = 2, sampling = False, delta = 0.1, eps = 0.1, subGSampling = False):
        '''Feature-based HAWE.'''
        self.embeddings = []
        numSamples = float("inf")
        if subGSampling:
            if not hasattr(self, 'subGs'):
                self.generate_subgraphs(hop)
        if length not in self.APHAW or len(self.APHAW[length]) <= 0:
            self.ap_HAW(length)
        startTime = time.time()
        for i in tqdm(range(self.numNodes)):
            if subGSampling:
                subG = self.subGs[i]
            else:
                subG = self.HG
            if not sampling:
                patterns = self.feature_based_subgraph_embedding_nonsampling(subG, length)
            else:
                numAPHAW = len(self.APHAW)
                numSamples = int(2*(math.log(2)*numAPHAW + math.log(1./delta))/eps**2)+1
                patterns = self.feature_based_subgraph_embedding_sampling(subG, length, numSamples)
            embedding = []
            for haw in self.APHAW[length]:
                embedding.append(patterns.get(tuple(haw), 0))
            self.embeddings.append(embedding)
        endTime = time.time()
        print('Embedding Computation is done. The whole time spent is {}s.'.format(endTime-startTime))
        print('Sample Number: {} ---- inf for nonsampling'.format(numSamples))
        self.embeddings = np.array(self.embeddings)
        return self.embeddings

    def generate_walk_paragraph(self, node, numWalksPerNode, length, subGSampling = False, method = 'H'):
        walkParagraph = []
        if subGSampling:
            givenRwG = self.create_random_walk_graph(subG)
        else:
            givenRwG = self.RwG
        #for node in subG.nodes():
        for _ in range(numWalksPerNode):
            if method == 'CH':
                haw = self.coarse_hetero_anonymous_walk(givenRwG, node, length)
            elif method == 'H':
                haw = self.hetero_anonymous_walk(givenRwG, node, length)
            walkParagraph.append(haw)
        return walkParagraph
            
    def generating_examples(self, contextSize, noise, numNegativeSamples, vocab,  workers):
        t0 = time.time()
        examples = []
        partialNodeIdsList = partition(list(range(self.numNodes)), workers)
        futures = {}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            part = 0
            for partialNodeIds in partialNodeIdsList:
                job = executor.submit(self.generating_examples_worker_func, partialNodeIds, contextSize, noise, numNegativeSamples, vocab)
                futures[job] = part
                part += 1
            for job in as_completed(futures):
                partialexamples = job.result()
                examples += partialexamples
        t1 = time.time()
        print("Consumed time for generating examples:",t1-t0)
        return examples

    def generating_examples_worker_func(self, idList, contextSize, noise, numNegativeSamples, vocab):
        partialExamples = []        
        for nodeId in idList:
            para = self.cleanedCorpus[nodeId]
            for i in range(contextSize, len(para) - contextSize):
                positiveSample = vocab.word2idx[para[i]]
                sampleIds = noise.sample(numNegativeSamples).tolist()
                sampleIds = [sampleId if sampleId != positiveSample else -1 for sampleId in sampleIds]
                sampleIds.insert(0, positiveSample) 
                context = para[i - contextSize:i] + para[i + 1:i + contextSize + 1]
                contextIds = [vocab.word2idx[w] for w in context]
                partialExamples.append( {"doc_ids":nodeId,
                       "sample_ids": sampleIds,
                       "context_ids": contextIds})
        return partialExamples

    def train(self, model, dataLoader, epochs, lr):
        optimizer = Adam(model.parameters(), lr=lr)
        training_losses = []
        loss = NegativeSampling()
        try:
            for epoch in trange(epochs, desc="Epochs"):
                epoch_losses = []
                for batch in dataLoader:
                    model.zero_grad()
                    logits = model.forward(**batch)
                    batch_loss = loss(logits)
                    epoch_losses.append(batch_loss.item())
                    batch_loss.backward()
                    optimizer.step()
                training_losses.append(np.mean(epoch_losses))
        except KeyboardInterrupt:
            print(f"Interrupted on epoch {epoch}!")
        finally:
            return training_losses

    def generate_walk_corpus_worker_func(self, idList, numWalksPerNode, length, subGSampling, method):
        t0 = time.time()
        partialCorpus = []
        j = 0
        for i in idList:
            partialCorpus.append(self.generate_walk_paragraph(i, numWalksPerNode, length, subGSampling, method))
        t1 = time.time()
        #print(t1-t0)
        return partialCorpus

    def generate_walk_corpus(self, length = 8, numWalksPerNode = 10, hop = 2, minCount = 1, workers=8, subGSampling =False, method = 'H'):
        if subGSampling:
            if not hasattr(self, 'subGs'):
                self.generate_subgraphs(hop)
        else:
            self.RwG = self.create_random_walk_graph(self.HG)
        '''if length not in self.APHAW or len(self.APHAW[length]) <= 0:
            self.ap_HAW(length)
        self.walk2id = {tuple(w): i for i, w in enumerate(self.APHAW[length])}'''
        print("Generating Walk Corpus")
        walkCorpus = []
        partialNodeIdsList = partition(list(range(self.numNodes)), workers)
        futures = {}
        t0 = time.time()
        results = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            part = 0
            for partialNodeIds in partialNodeIdsList:
                job = executor.submit(self.generate_walk_corpus_worker_func, partialNodeIds, numWalksPerNode, length, subGSampling, method)
                futures[job] = part
                part += 1
            for job in as_completed(futures):
                partialCorpus = job.result()
                #print(futures[job])
                results[futures[job]] = partialCorpus
        for i in range(workers):
            walkCorpus += results[i]
        #self.generate_walk_paragraph(i, self.subGs[i], numWalksPerNode, length)
        allTokens = [word for para in walkCorpus for word in para]
        self.vocab = Vocab(allTokens, min_count=minCount)
        clean_paragraph = lambda x: [t for t in x if t in self.vocab.freqs.keys()]
        self.cleanedCorpus = [clean_paragraph(para) for para in walkCorpus]
        t1 = time.time()
        print("Consumed time for generating corpus:",t1-t0)

    def create_walk_dataloader(self, contextSize = 5, numNegativeSamples = 5, batch_size = 50, workers = 8):
        print("Create Walk Dataloader")
        t0 = time.time()
        noise = NoiseDistribution(self.vocab)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        examples = self.generating_examples(contextSize= contextSize, noise = noise, numNegativeSamples = numNegativeSamples, vocab = self.vocab, workers = workers)
        print(time.time()-t0)
        examples = [{"doc_ids": torch.tensor(e["doc_ids"]).to(device),
                       "sample_ids": torch.tensor(e["sample_ids"]).to(device),
                       "context_ids": torch.tensor(e["context_ids"]).to(device)} for e in examples]
        docDataset = NCEDataset(examples)
        self.dataLoader = DataLoader(docDataset, batch_size = batch_size, drop_last = True, shuffle=True) 
        t1 = time.time()
        print("Consumed time for generating corpus:",t1-t0)

    def distributed_memory_subgraph_embedding(self, length = 8, numWalksPerNode = 10, hop = 2, minCount = 1, workers = 8,# parameters for corpus generation 
            contextSize = 5, numNegativeSamples = 5,  batch_size = 50, # parameters for processing data 
            dim = 128, epochs = 40, lr = 1e-3,  # parameters for training 
            skip = True): # if skip = true, do not regenerate corpus and examples
        if not hasattr(self, 'cleanedCorpus') or skip == False:
            self.generate_walk_corpus(length = length, numWalksPerNode = numWalksPerNode, hop = hop, minCount = minCount, workers = workers)
        if not hasattr(self, 'dataLoader') or skip == False:
            self.create_walk_dataloader(contextSize = contextSize, numNegativeSamples = numNegativeSamples, batch_size=batch_size, workers = workers)
        print('# words', len(self.vocab.words))
        print('# examples', len(self.dataLoader))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        self.model = DistributedMemory(vec_dim=dim, n_docs=self.numNodes, n_words=len(self.vocab.words)).to(device)
        t0 = time.time()
        training_losses = self.train(self.model, self.dataLoader, epochs, lr)
        print('training time:', time.time()-t0)
        self.embeddings = self.model.paragraph_matrix.cpu().data.numpy()
        return self.embeddings, training_losses
    
    def gensim_doc2vec_examples(self, degreeTagged= False):
        return LabeledDocuments(self.HG, self.cleanedCorpus, self.vocab, self.NTL, degreeTagged)


class LabeledDocuments(object):
    def __init__(self, HG, cleanedCorpus, vocab, NTL, degreeTagged = False):
        self.cleanedCorpus = cleanedCorpus
        self.vocab = vocab
        self.NTL = NTL
        self.HG = HG
        self.dtag = degreeTagged
        if self.dtag:
            self.dglist = ['Deg'+str(int(math.log(self.HG.degree(node)+1,2))) for node in range(len(self.HG.nodes()))]
        
    def __iter__(self):
        for nodeId,para in enumerate(self.cleanedCorpus):
            if self.dtag:
                yield LabeledSentence([ str(self.vocab.word2idx[sent]) for sent in para], [self.NTL[nodeId]+str(nodeId), self.dglist[nodeId]])
            else:
                yield LabeledSentence([ str(self.vocab.word2idx[sent]) for sent in para], [self.NTL[nodeId]+str(nodeId)])

import torch.nn as nn        
from torch.utils.data import Dataset, DataLoader

class Vocab:
    def __init__(self, all_tokens, min_count=2):
        self.min_count = min_count
        self.freqs = {t:n for t, n in Counter(all_tokens).items() if n >= min_count}
        self.words = sorted(self.freqs.keys())
        self.word2idx = {w: i for i, w in enumerate(self.words)}

class NoiseDistribution:
    def __init__(self, vocab):
        self.probs = np.array([vocab.freqs[w] for w in vocab.words])
        self.probs = np.power(self.probs, 0.75)
        self.probs /= np.sum(self.probs)
    def sample(self, n):
        "Returns the indices of n words randomly sampled from the vocabulary."
        return np.random.choice(a=self.probs.shape[0], size=n, p=self.probs)

class NCEDataset(Dataset):
    def __init__(self, examples):
        self.examples = list(examples)  
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, index):
        return self.examples[index]

class NegativeSampling(nn.Module):
    def __init__(self):
        super(NegativeSampling, self).__init__()
        self.log_sigmoid = nn.LogSigmoid()
    def forward(self, scores):
        batch_size = scores.shape[0]
        n_negative_samples = scores.shape[1] - 1  
        positive = self.log_sigmoid(scores[:,0])
        negatives = torch.sum(self.log_sigmoid(-scores[:,1:]), dim=1)
        return -torch.sum(positive + negatives) / batch_size  

class DistributedMemory(nn.Module):
    def __init__(self, vec_dim, n_docs, n_words):
        super(DistributedMemory, self).__init__()
        self.paragraph_matrix = nn.Parameter(torch.randn(n_docs, vec_dim))
        self.word_matrix = nn.Parameter(torch.randn(n_words, vec_dim))
        self.outputs = nn.Parameter(torch.zeros(vec_dim , n_words))
    
    def forward(self, doc_ids, context_ids, sample_ids):   
        inputs = torch.add(self.paragraph_matrix[doc_ids,:],  torch.mean(self.word_matrix[context_ids,:], dim=1))  
        #imputs = torch
        outputs = self.outputs[:,sample_ids]                                 
        return torch.bmm(inputs.unsqueeze(dim=1), outputs.permute(1, 0, 2)).squeeze()                   
