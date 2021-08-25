from .packages import *

class Network:
    def __init__(self, net, loss=None, optimizer=None, gamma=1.0, path=None):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        if optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                    gamma=gamma)
        self.path = path
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def predict(self, x):
        return self.net(x)
    
    def predict_from_loader(self, loader, to_numpy=False):
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                preds.append(self.predict(x))
                targets.append(y)
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        if to_numpy:
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()
        return preds, targets
    
    def train(self, train_loader, test_loader=None, epochs=1, metrics=[], early_stop=float('inf'), verbose=True):
        train_early_stopping = EarlyStopping(early_stop)
        test_early_stopping = EarlyStopping(early_stop)
        metrics = set(['loss'] + metrics)
        history = collections.defaultdict(list)
        for epoch in range(1, epochs+1):
            # Training steps
            history['epoch'].append(epoch)
            for key, val in self.train_epoch(train_loader, metrics).items():
                history['train_'+key].append(val)
            
            # Eval on test data
            if test_loader is not None:
                for key, val in self.evaluate(test_loader, metrics).items():
                    history['test_'+key].append(val)
                
            # Print metrics
            if verbose:
                to_print = ''
                for key, val in history.items():
                    to_print += f'{key}: {np.round(val[-1], 6)}, '
                print(to_print)
            
            # Early stopping and save best only
            if train_early_stopping.should_stop(history['train_loss'][-1]):
                break
            if test_loader is not None:
                test_loss = history['test_loss'][-1]
                if test_early_stopping.is_improvement(test_loss):
                    self.best_loss = test_loss
                    if self.path is not None:
                        self.save(self.path)
                if test_early_stopping.should_stop(test_loss):
                    break
                    
        # Save or load best model
        if self.path is not None:
            if test_loader is not None:
                self.load(self.path)
            else:
                self.save(self.path)
                
        # Turn eval mode and return history
        self.net.eval()
        history = pd.DataFrame(history)
        history = history.set_index('epoch')
        return history
    
    def train_epoch(self, train_loader, metrics=['loss']):
        self.net.train()
        n = 0
        results = collections.defaultdict(lambda : 0)
        for x, y in train_loader:
            self.optimizer.zero_grad()
            preds = self.predict(x)
            loss = self.loss(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.net.parameters(), 1.0)
            self.optimizer.step()
            for key, val in self.get_metrics(preds, y, metrics).items():
                results[key] += val * len(preds)
            n += len(preds)
        self.scheduler.step()
        for key in results:
            results[key] /= n
        return results
    
    def evaluate(self, test_loader, metrics=['loss']):
        self.net.eval()
        n = 0
        results = collections.defaultdict(lambda : 0)
        with torch.no_grad():
            for x, y in test_loader:
                preds = self.predict(x)
                for key, val in self.get_metrics(preds, y, metrics).items():
                    results[key] += val * len(preds)
                n += len(preds)
                    
        for key in results:
            results[key] /= n
        return results
    
    def get_metrics(self, preds, y, metrics):
        results = {}
        if 'loss' in metrics:
            results['loss'] = self.get_loss(preds, y)
        if 'acc' in metrics:
            results['acc'] = self.get_acc(preds, y)
        return results
    
    def get_loss(self, preds, y):
        with torch.no_grad():
            score =  self.loss(preds, y).item()
        return score
    
    @staticmethod
    def get_acc(preds, y):
        with torch.no_grad():
            if preds.shape[1] > 1:
                preds = preds.argmax(1, keepdims=True)
            else:
                preds = preds.round()
            score = ((preds==y).sum() / y.shape[0]).item()
        return score
    
    def save(self, file_name):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }, file_name)
    
    def load(self, file_name, device=None):
        if device is None:
            checkpoint = torch.load(file_name)
        else:
            checkpoint = torch.load(file_name, map_location=torch.device(device))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss']
        
        
class EarlyStopping:
    def __init__(self, patience, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def should_stop(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss < self.best_loss + self.delta:
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            return True
        return False
    
    def is_improvement(self, val_loss):
        if val_loss < self.best_loss:
            return True
        return False
        
    
class StackingEnsemble:
    def __init__(self, models, weighter=RidgeCV()):
        self.models = models
        self.weighter = weighter
        
    def train(self, train_loader, test_loader=None,
              epochs=1, early_stop=5, metrics=[], verbose=False):
        # Train Models
        for i, model in enumerate(self.models):
            if verbose:
                print('-'*5, i, '-'*5)
            model.train(train_loader, test_loader=test_loader, epochs=epochs, 
                        early_stop=early_stop,  metrics=metrics, verbose=verbose)
        # Train weighter
        experts, targets = self.get_experts(train_loader)
        self.weighter.fit(experts, targets)
        
    def get_experts(self, loader):
        experts = [[] for i in range(len(self.models))]
        targets = []
        with torch.no_grad():
            for x, y in loader:
                for i, model in enumerate(self.models):
                    experts[i].append(model.predict(x))
                targets.append(y)
        experts = [torch.cat(exp).cpu().numpy() for exp in experts]
        targets = torch.cat(targets).cpu().numpy()
        return np.hstack(experts), targets
    
    def predict(self, loader):
        experts, targets = self.get_experts(loader)
        preds = self.weighter.predict(experts)
        return preds, targets
    
    
class VotingEnsemble:
    def __init__(self, models):
        self.models = models
        
    def train(self, train_loader, test_loader=None,
              epochs=1, early_stop=5, metrics=[], verbose=False):
        # Train Models
        history = {}
        for i, model in enumerate(self.models):
            if verbose:
                print('-'*5, i, '-'*5)
            history[f'net{i}'] = model.train(train_loader, test_loader=test_loader,
                                             epochs=epochs, early_stop=early_stop,
                                             metrics=metrics, verbose=verbose)
        return history
    
    def get_experts(self, loader):
        experts = [[] for i in range(len(self.models))]
        targets = []
        with torch.no_grad():
            for x, y in loader:
                for i, model in enumerate(self.models):
                    experts[i].append(model.predict(x))
                targets.append(y)
        experts = [torch.cat(exp).cpu().numpy() for exp in experts]
        targets = torch.cat(targets).cpu().numpy()
        return np.hstack(experts), targets
    
    def predict_from_loader(self, loader, to_numpy=None):
        experts, targets = self.get_experts(loader)
        preds = experts.mean(1)
        return preds, targets
    

class RecurrentEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, num_layers,
                 rnn_type='gru', pool_type='last', bidirectional=False, dropout=0):
        super().__init__()
        
        if rnn_type == 'rnn':
            RNN = nn.RNN
        elif rnn_type == 'lstm':
            RNN = nn.LSTM
        else:
            RNN = nn.GRU
            
        if pool_type == 'max':
            pool = self.packed_maxpool
        elif pool_type == 'avg':
            pool = self.packed_avgpool
        elif bidirectional:
            pool = self.packed_bilastpool
        else:
            pool = self.packed_lastpool
            
        if bidirectional:
            hidden_dim = output_dim // 2
        else:
            hidden_dim = output_dim

        self.embeddings = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = RNN(embedding_dim, hidden_dim, num_layers,
                       dropout=dropout, bidirectional=bidirectional)
        self.pool = pool
        
    def forward(self, x_tuple):
        x, lens = x_tuple
        out = self.embeddings(x)
        out = self.dropout(out)
        out = nn.utils.rnn.pack_padded_sequence(out, lens, enforce_sorted=False)
        out, _ = self.rnn(out)
        out = self.pool(out)
        return out
    
    @staticmethod
    def packed_lastpool(x_packed):
        out, lens = nn.utils.rnn.pad_packed_sequence(x_packed)
        return torch.vstack([out[l-1, i] for i, l in enumerate(lens)])
    
    @staticmethod
    def packed_bilastpool(x_packed):
        out, lens = nn.utils.rnn.pad_packed_sequence(x_packed)
        hidden_dim = out.shape[-1] // 2
        out1 = torch.vstack([out[l-1, i, :hidden_dim] for i, l in enumerate(lens)])
        out2 = out[0,:,hidden_dim:]
        out = torch.cat([out1, out2], -1)
        return out
    
    @staticmethod
    def packed_avgpool(x_packed):
        out, lens = nn.utils.rnn.pad_packed_sequence(x_packed, padding_value=0)
        return out.sum(0) / lens.view(-1, 1).to(device)
    
    @staticmethod
    def packed_maxpool(x_packed):
        out, _ = nn.utils.rnn.pad_packed_sequence(x_packed, padding_value=float('-inf'))
        return out.max(0)[0]
    

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kind='bert',
                 num_heads=1, num_layers=1, dropout=0):
        super().__init__()
        self.kind = kind
        self.output_dim = output_dim
        self.embeddings = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(output_dim)
        self.dropout = nn.Dropout(dropout)
        transformer_layer = nn.TransformerEncoderLayer(output_dim, num_heads, hidden_dim, dropout)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
    
    def forward(self, x_tuple):
        x, lens = x_tuple
        padding_mask = self.get_padding_mask(x)
        if self.kind == 'gpt':
            src_mask = self.get_src_mask(x.shape[0])
        else: 
            src_mask = None
        out = self.embeddings(x)
        out = self.pos_encoder(out * np.sqrt(self.output_dim))
        out = self.dropout(out)
        out = self.transformer(out, src_mask, padding_mask)
        return self.avgpool(out, lens)
    
    @staticmethod
    def get_src_mask(n):
        mask = (torch.triu(torch.ones(n, n))==1).T
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, 0.0)
        return mask
    
    @staticmethod
    def get_padding_mask(x):
        return x.T == 0
    
    @staticmethod
    def avgpool(x, lens):
        return torch.vstack([x[:l-1, i].mean(0) for i, l in enumerate(lens)])
        
        
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2, dtype=torch.float)
                             * (-np.log(10000) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
    
class TransformerRecurrentEncoder(TransformerEncoder, RecurrentEncoder):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1,
                 num_layers=1, dropout=0, bidirectional=False):
        nn.Module.__init__(self)
        self.output_dim = output_dim
        
        if bidirectional:
            rnn_hidden_dim = output_dim // 2
        else:
            rnn_hidden_dim = output_dim
            
        self.embeddings = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(output_dim)
        self.dropout = nn.Dropout(dropout)
        transformer_layer = nn.TransformerEncoderLayer(output_dim, num_heads, hidden_dim, dropout)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
        self.rnn = nn.GRU(output_dim, rnn_hidden_dim, num_layers,
                          dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, x_tuple):
        x, lens = x_tuple
        padding_mask = self.get_padding_mask(x)
        out = self.embeddings(x)
        out = self.pos_encoder(out * np.sqrt(self.output_dim))
        out = self.dropout(out)
        out = self.transformer(out, None, padding_mask)
        out[padding_mask.T] = 0
        out = self.dropout(out)
        out = nn.utils.rnn.pack_padded_sequence(out, lens, enforce_sorted=False)
        out, _ = self.rnn(out)
        return self.packed_avgpool(out)


class RecurrentTransformerEncoder(TransformerEncoder, RecurrentEncoder):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1,
                 num_layers=1, dropout=0, bidirectional=False):
        nn.Module.__init__(self)
        self.output_dim = output_dim
        
        if bidirectional:
            rnn_hidden_dim = output_dim // 2
        else:
            rnn_hidden_dim = output_dim
            
        self.embeddings = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(output_dim, rnn_hidden_dim, num_layers,
                          dropout=dropout, bidirectional=bidirectional)
        transformer_layer = nn.TransformerEncoderLayer(output_dim, num_heads, hidden_dim, dropout)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
    
    def forward(self, x_tuple):
        x, lens = x_tuple
        padding_mask = self.get_padding_mask(x)
        out = self.embeddings(x)
        out = self.dropout(out)
        out[padding_mask.T] = 0
        out = nn.utils.rnn.pack_padded_sequence(out, lens, enforce_sorted=False)
        out, _ = self.rnn(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        out = self.dropout(out)
        out = self.transformer(out, None, padding_mask)
        return self.avgpool(out, lens)


class ParallelRTEncoder(TransformerEncoder, RecurrentEncoder):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1,
                 num_layers=1, dropout=0, bidirectional=False):
        nn.Module.__init__(self)
        output_dim = output_dim // 2 
        self.output_dim = output_dim
        
        if bidirectional:
            rnn_hidden_dim = output_dim // 2
        else:
            rnn_hidden_dim = output_dim
            
        self.embeddings = nn.Embedding(input_dim, output_dim, padding_idx=0)
        
        self.rnn = nn.GRU(output_dim, rnn_hidden_dim, num_layers,
                          dropout=dropout, bidirectional=bidirectional)
        
        self.pos_encoder = PositionalEncoding(output_dim)
        transformer_layer = nn.TransformerEncoderLayer(output_dim, num_heads, hidden_dim, dropout)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
    
    def forward(self, x_tuple):
        x, lens = x_tuple
        out = self.embeddings(x)

        # RNN
        rnn_out = nn.utils.rnn.pack_padded_sequence(out, lens, enforce_sorted=False)
        rnn_out, _ = self.rnn(rnn_out)
        rnn_out = self.packed_avgpool(rnn_out)

        # Transformer
        padding_mask = self.get_padding_mask(x)
        trans_out = self.pos_encoder(out * np.sqrt(self.output_dim))
        trans_out = self.transformer(trans_out, None, padding_mask)
        trans_out = self.avgpool(trans_out, lens)

        return torch.hstack([rnn_out, trans_out])
        
        
class ConvolutionalEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, dropout=0):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        out_channels = output_dim // 2
        for kernel_size in [1, 3]:
            padding_size =  int((kernel_size - 1) // 2)
            conv = nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=padding_size)
            self.convs.append(conv)
        self.relu = nn.ReLU()
        
    def forward(self, x_tuple):
        x, lens = x_tuple
        out = torch.nn.functional.one_hot(x, self.input_dim).permute(1, 2, 0).float()
        out = self.dropout(out)
        out = [conv(out) for conv in self.convs]
        out = torch.cat(out, 1).permute(2, 0, 1)
        out = self.ReLU(out)
        return self.avgpool(out, lens)
    
    @staticmethod
    def avgpool(x, lens):
        return torch.vstack([x[:l-1, i].mean(0) for i, l in enumerate(lens)])
        
        
        for kernel_size, stride in zip(kernel_sizes, stride_sizes):
            padding_size =  int((kernel_size - 1) // 2)
            conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding_size)
            self.convs.append(conv)
            
            
class ConvolutionalRecurrentEncoder(RecurrentEncoder, ConvolutionalEncoder):
    def __init__(self, input_dim, output_dim, dropout=0, num_layers=1, bidirectional=False):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.embeddings = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.convs = nn.ModuleList()
        out_channels = output_dim // 2
        for kernel_size in [1, 3]:
            padding_size =  int((kernel_size - 1) // 2)
            conv = nn.Conv1d(in_channels=input_dim, out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=padding_size)
            self.convs.append(conv)
        self.relu = nn.ReLU()
                    
        if bidirectional:
            rnn_hidden_dim = output_dim // 2
        else:
            rnn_hidden_dim = output_dim
            
        self.rnn = nn.GRU(output_dim, rnn_hidden_dim, num_layers,
                          dropout=dropout, bidirectional=bidirectional)
        
    def forward(self, x_tuple):
        x, lens = x_tuple
        out = torch.nn.functional.one_hot(x, self.input_dim).permute(1, 2, 0).float()
        out = self.dropout(out)
        
        out = [conv(out) for conv in self.convs]
        out = torch.cat(out, 1).permute(2, 0, 1)
        self.relu(out)
        out = self.dropout(out)
        
        out = nn.utils.rnn.pack_padded_sequence(out, lens, enforce_sorted=False)
        out, _ = self.rnn(out)
        out = self.packed_avgpool(out)
        return out

    
class SiameseNet(nn.Module):
    def __init__(self, encoder):
        super(SiameseNet, self).__init__()
        self.encoder = encoder
    
    def forward(self, x):
        return [self.encoder(xi) for xi in x]
    

class SupervisedSiameseNet(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
    
    def forward(self, x):
        embeds = [self.encoder(xi) for xi in x]
        preds = [self.predictor(embed) for embed in embeds]
        return (embeds, preds)


class PrallelEncoder(nn.Module):
    def __init__(self, *encoders):
        super().__init__()
        self.encoders = nn.ModuleList(list(encoders))

    def forward(self, x):
        outs = [encoder(x) for encoder in self.encoders]
        return torch.hstack(outs)
    
class FeaturesEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x_tuple):
        x, features = x_tuple
        out = self.encoder(x)
        return torch.hstack([out, features])