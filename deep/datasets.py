from .packages import *

class Vocabulary(dict):
    def __init__(self, tokens):
        super().__init__()
        tokens = ['<pad>', '<unk>'] + list(tokens)
        self.stoi = {token: i for i, token in enumerate(tokens)}
        self.itos = {i: token for i, token in enumerate(tokens)}
    
    def __getitem__(self, key):
        try:
            return self.stoi[key]
        except:
            return self.stoi['<unk>']
        
    def encode(self, seq):
        return [self[t] for t in seq]
    
    def decode(self, seq):
        return [self.itos[t] for t in seq]
        

class EmbedsDataset(Dataset):
    def __init__(self, peptides, targets, embeds_map, aminoacids):
        self.peptides = list(peptides)
        self.targets = list(targets)
        self.embeds_map = embeds_map
    
    def __len__(self):
        return len(self.peptides)
    
    def __getitem__(self, index):
        peptide = self.embeds_map[self.peptides[index]]
        target = self.targets[index]
        return torch.tensor(peptide), torch.tensor(target)


class PeptideDataset(Dataset):
    def __init__(self, peptides, targets, aminoacids):
        self.peptides = list(peptides)
        self.targets = list(targets)
        self.vocab = Vocabulary(aminoacids)
        
    def __len__(self):
        return len(self.peptides)
    
    def __getitem__(self, index):
        peptide = self.vocab.encode(self.peptides[index])
        target = self.targets[index]
        return torch.tensor(peptide), torch.tensor(target)
    

class Collate:
    def __init__(self, padding_value=1):
        self.padding_value = padding_value
        
    def __call__(self, batch):
        peptides, lens, targets = [], [], []
        for peptide, target in batch:
            peptides.append(peptide)
            lens.append(len(peptide))
            targets.append(target)
        peptides = nn.utils.rnn.pad_sequence(peptides,
                                             padding_value=self.padding_value)
        lens = torch.tensor(lens, dtype=torch.int)
        targets = torch.tensor(targets, dtype=torch.float)
        return (peptides.to(device), lens), targets.to(device).view(-1, 1)
    
    
###########
# Siamese #
###########
    
    
class PeptideSiameseDataset(PeptideDataset):
    def __init__(self, peptides, targets, n, aminoacids, data_generator=None):
        if data_generator is None:
            peptides, targets = self.generate_data(peptides, targets, int(n))
        else:
            peptides, targets = data_generator(peptides, targets, int(n))
        self.peptides = list(peptides)
        self.targets = list(targets)
        self.vocab = Vocabulary(aminoacids)
    
    def __getitem__(self, index):
        peptide1, peptide2 = self.peptides[index]
        peptide1 = self.vocab.encode(peptide1)
        peptide2 = self.vocab.encode(peptide2)
        target1, target2 = self.targets[index]
        return ((torch.tensor(peptide1), torch.tensor(peptide2)),
                (torch.tensor(target1), torch.tensor(target2)))
    
    def generate_data(self, peptides, targets, n):
        idxs1 = np.random.randint(targets.size, size=n)
        idxs2 = np.random.randint(targets.size, size=n)
        peptides = [peptides[idxs1], peptides[idxs2]]
        peptides = np.vstack(peptides).T
        targets = [targets[idxs1], targets[idxs2]]
        targets = np.vstack(targets).T
        return peptides, targets
    
    
class CollateSiamese(Collate):
    def __init__(self, similarity, padding_value=0):
        self.similarity = similarity
        self.padding_value = padding_value
    
    def __call__(self, batch):
        peptides1, peptides2 = [], []
        lens1, lens2 = [], []
        similarities = []
        for (peptide1, peptide2), (target1, target2) in batch:
            peptides1.append(peptide1)
            peptides2.append(peptide2)
            lens1.append(len(peptide1))
            lens2.append(len(peptide2))
            similarities.append(self.similarity(target1, target2))
        peptides1 = nn.utils.rnn.pad_sequence(peptides1,
                                              padding_value=self.padding_value)
        peptides2 = nn.utils.rnn.pad_sequence(peptides2,
                                              padding_value=self.padding_value)
        similarities = torch.tensor(similarities, dtype=torch.float).view(-1, 1)
        return ((peptides1.to(device), lens1), (peptides2.to(device), lens2)), similarities.to(device)
    
class CollateSupervisedSiamese(Collate):
    def __init__(self, similarity, padding_value=0):
        self.similarity = similarity
        self.padding_value = padding_value
    
    def __call__(self, batch):
        peptides1, peptides2 = [], []
        lens1, lens2 = [], []
        targets1, targets2 = [], []
        similarities = []
        for (peptide1, peptide2), (target1, target2) in batch:
            peptides1.append(peptide1)
            peptides2.append(peptide2)
            lens1.append(len(peptide1))
            lens2.append(len(peptide2))
            targets1.append(target1)
            targets2.append(target2)
            similarities.append(self.similarity(target1, target2))
        peptides1 = nn.utils.rnn.pad_sequence(peptides1,
                                              padding_value=self.padding_value)
        peptides2 = nn.utils.rnn.pad_sequence(peptides2,
                                              padding_value=self.padding_value)
        similarities = torch.tensor(similarities, dtype=torch.float32).view(-1, 1)
        targets1 = torch.tensor(targets1, dtype=torch.float).view(-1, 1)
        targets2 = torch.tensor(targets2, dtype=torch.float).view(-1, 1)
        return (((peptides1.to(device), lens1), (peptides2.to(device), lens2)),
                (similarities.to(device), (targets1.to(device), targets2.to(device))))

    
###########    
# Triplet #    
###########

    
class PeptideTripletDataset(PeptideSiameseDataset):
    def __init__(self, peptides, targets, n, aminoacids, data_generator=None):
        peptides, targets = self.generate_data(peptides, targets, int(n))
        self.peptides = list(peptides)
        self.targets = list(targets)
        self.vocab = Vocabulary(aminoacids)
    
#     def __getitem__(self, index):
#         peptide1, peptide2, peptide3 = self.peptides[index]
#         peptide1 = self.vocab.encode(peptide1)
#         peptide2 = self.vocab.encode(peptide2)
#         peptide3 = self.vocab.encode(peptide3)
#         target1, target2, target3 = self.targets[index]
#         return torch.tensor(peptide1), torch.tensor(peptide2), torch.tensor(peptide3)
    
    def __getitem__(self, index):
        peptides = []
        for peptide in self.peptides[index]:
            peptides.append(torch.tensor(self.vocab.encode(peptide)))
        targets = []
        for target in self.targets[index]:
            targets.append(torch.tensor(target))
        return peptides, targets
    
    def generate_data(self, peptides, targets, n):
        n_pos = n//2
        n_neg = n - n_pos
        idxs_pos = np.where(targets==1)[0]
        idxs_neg = np.where(targets==0)[0]
        # Positive anchors
        idxs_pa = np.random.choice(idxs_pos, size=n_pos)
        idxs_pp = np.random.choice(idxs_pos, size=n_pos)
        idxs_pn = np.random.choice(idxs_neg, size=n_pos)
        # Negative anchors
        idxs_na = np.random.choice(idxs_neg, size=n_neg)
        idxs_np = np.random.choice(idxs_neg, size=n_neg)
        idxs_nn = np.random.choice(idxs_pos, size=n_neg)
        # Combine
        idxs_a = np.append(idxs_pa, idxs_na)
        idxs_p = np.append(idxs_pp, idxs_pa)
        idxs_n = np.append(idxs_pn, idxs_nn)
        peptides = [peptides[idxs_a], peptides[idxs_p], peptides[idxs_n]]
        peptides = np.vstack(peptides).T
        targets = [targets[idxs_a], targets[idxs_p], targets[idxs_n]]
        targets = np.vstack(targets).T
        return peptides, targets
    
    
class CollateTriplet(Collate):
    def __init__(self, padding_value=0):
        self.padding_value = padding_value
    
    def __call__(self, batch):
        peptides_dict = collections.defaultdict(list)
        lens_dict = collections.defaultdict(list)
        for peptides, targets in batch:
            for i, peptide in enumerate(peptides):
                peptides_dict[i].append(peptide)
                lens_dict[i].append(len(peptide))
        x = []
        for i in peptides_dict:
            peptide = nn.utils.rnn.pad_sequence(
                peptides_dict[i], padding_value=self.padding_value
            ).to(device)
            lens = lens_dict[i]
            x.append((peptide, lens))
        return x, None
    
    
class CollateSupervisedTriplet(Collate):
    def __init__(self, padding_value=0):
        self.padding_value = padding_value
    
    def __call__(self, batch):
        peptides_dict = collections.defaultdict(list)
        targets_dict = collections.defaultdict(list)
        lens_dict = collections.defaultdict(list)
        for peptides, targets in batch:
            for i, (peptide, target) in enumerate(zip(peptides, targets)):
                peptides_dict[i].append(peptide)
                lens_dict[i].append(len(peptide))
                targets_dict[i].append(target)
        x = []
        for i in peptides_dict:
            peptide = nn.utils.rnn.pad_sequence(
                peptides_dict[i], padding_value=self.padding_value
            ).to(device)
            lens = lens_dict[i]
            x.append((peptide, lens))
        y = []
        for targets in targets_dict.values():
            y.append(torch.FloatTensor(targets).view(-1, 1).to(device))
        return x, y
    
    
###########
# NesxtAA #
###########

    
class PeptideNextAADataset(PeptideDataset):
    def __init__(self, peptides, aminoacids):
        self.peptides = list(peptides)
        self.vocab = Vocabulary(aminoacids)
    
    def __getitem__(self, index):
        peptide = self.vocab.encode(self.peptides[index])
        return torch.tensor(peptide)
    
    
class CollateNextAA(Collate):
    def __call__(self, batch):
        peptides, targets = [], []
        for peptide in batch:
            peptides.append(peptide[:-1])
            targets.append(peptide[-1])
        peptides = nn.utils.rnn.pad_sequence(peptides,
                                             padding_value=self.padding_value)
        targets = torch.tensor(targets, dtype=torch.int) - 2
        return peptides.to(device), targets.to(device).view(-1, 1)
    
    
#################
# RetensionTime #
#################
    
    
class PeptideFeaturesDataset(PeptideDataset):
    def __init__(self, x, targets, aminoacids):
        self.peptides = list(x[:,0])
        self.features = x[:,1:].tolist()
        self.targets = list(targets)
        self.vocab = Vocabulary(aminoacids)
        
    def __len__(self):
        return len(self.peptides)
    
    def __getitem__(self, index):
        peptide = self.vocab.encode(self.peptides[index])
        feature = self.features[index]
        target = self.targets[index]
        return torch.tensor(peptide), torch.tensor(feature), torch.tensor(target)
    

class CollateFeatures:
    def __init__(self, padding_value=1):
        self.padding_value = padding_value
        
    def __call__(self, batch):
        peptides, lens, features, targets = [], [], [], []
        for peptide, feature, target in batch:
            peptides.append(peptide)
            lens.append(len(peptide))
            features.append(feature)
            targets.append(target)
        peptides = nn.utils.rnn.pad_sequence(peptides,
                                             padding_value=self.padding_value)
        lens = torch.tensor(lens, dtype=torch.int)
        features = torch.vstack(features).float()
        targets = torch.tensor(targets, dtype=torch.float)
        return ((peptides.to(device), lens), features.to(device)), targets.to(device).view(-1, 1)