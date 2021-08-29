from .packages import *
from .datasets import *


def all_to_one():
    files = ['packages.py', 'utils.py', 'losses.py', 'datasets.py', 'models.py']
    megafile = ''
    for i, file in enumerate(files):
        megafile += f'#####---{file}---#####\n\n'
        start = 0
        if i > 0:
            start = 2
        with open(file, 'r') as f:
            for line in f.readlines()[start:]:
                megafile += line
        megafile += '\n\n\n'
    with open('all.txt', 'w') as f:
        f.write(megafile)

def get_synthetic_data(n, aminoacids):
    lens = np.random.lognormal(mean=3, sigma=0.3, size=n).astype(int)
    peptides = []
    for l in lens:
        peptide = list(np.random.choice(aminoacids, size=l))
        peptide = ''.join(peptide)
        peptides.append(peptide)
    reproducibilities = 0.5 + 0.2*np.random.randn(n)
    reproducibilities[reproducibilities>1] = 0.4 + 0.1*np.random.randn()
    reproducibilities[reproducibilities<-0.2] = 0.4 + 0.1*np.random.randn()

    df = pd.DataFrame({'peptide':peptides, 'reproducibility':reproducibilities})
    return df

def get_siamese_data(peptides, targets, n):
    data_size = targets.size
    idxs = np.argsort(targets)
    peptides, targets = peptides[idxs], targets[idxs]
    idxs = np.arange(data_size)
    probs = abs(idxs - idxs[idxs[-1]//2])**2 + data_size
    probs = probs / probs.sum()
    idxs1 = np.random.choice(idxs, size=n, p=probs)
    idxs2 = np.random.choice(idxs, size=n, p=probs)
    peptides = [peptides[idxs1], peptides[idxs2]]
    peptides = np.vstack(peptides).T
    targets = [targets[idxs1], targets[idxs2]]
    targets = np.vstack(targets).T
    return peptides, targets

def get_embeddings(encoder, loader):
    with torch.no_grad():
        embeds, targets = [], []
        for peptides, y in loader:
            embeds.append(encoder(peptides))
            targets.append(y)
        embeds = torch.cat(embeds).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()
    return embeds, targets

def test_encoder(encoder, x_train, y_train, x_test, y_test):
    train_loader, test_loader = get_loaders(x_train, y_train, x_test, y_test,
                                            PeptideDataset, Collate(), 1024)
    train_embeds, train_targets = get_embeddings(encoder, train_loader)
    test_embeds, test_targets = get_embeddings(encoder, test_loader)
    
    clf = LogisticRegressionCV(solver='sag').fit(train_embeds, train_targets)
    train_preds = clf.predict_proba(train_embeds)[:,1]
    test_preds = clf.predict_proba(test_embeds)[:,1]
    
    metrics = {
        'BCE': log_loss,
        'ACC': lambda targets, preds: accuracy_score(targets, np.round(preds).astype(int)),
        'AUC': roc_auc_score,
        'Confusion': lambda targets, preds: confusion_matrix(targets, np.round(preds).astype(int)).ravel(),
    }
    results = {}
    for k, v in metrics.items():
        results['Train '+k] = v(train_targets, train_preds)
        results['Test '+k] = v(test_targets, test_preds)
    knn = KNeighborsClassifier().fit(train_embeds, train_targets)
    results['KNN ACC'] = knn.score(test_embeds, test_targets)
    return results

def test_classifier(clf, x_train, y_train, x_test, y_test):
    train_loader, test_loader = get_loaders(x_train, y_train, x_test, y_test,
                                            PeptideDataset, Collate(), 1024)
    train_preds, train_targets = clf.predict_from_loader(train_loader, to_numpy=True)
    test_preds, test_targets = clf.predict_from_loader(test_loader, to_numpy=True)
    metrics = {
        'BCE': log_loss,
        'ACC': lambda targets, preds: accuracy_score(targets, np.round(preds).astype(int)),
        'AUC': roc_auc_score,
        'Confusion': lambda targets, preds: confusion_matrix(targets, np.round(preds).astype(int)).ravel()
    }
    results = {}
    for k, v in metrics.items():
        results['Train '+k] = v(train_targets, train_preds)
        results['Test '+k] = v(test_targets, test_preds)
    return results

def test_embeddings(encoder, models, loader, 
                    normalize=False, scoring=None, cv=10):
    embeds, targets = get_embeddings(encoder, loader)
    if normalize:
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    results = {'Model':[], 'Score':[]}
    for model_name, model in models:
        scores = cross_val_score(model, embeds, targets,
                                 cv=10, scoring=scoring, n_jobs=-1)
        results['Score'] += list(scores)
        results['Model'] += [model_name] * len(scores)
        
    results = pd.DataFrame(results)
    
    sns.boxplot(y='Model', x='Score', data=results,
                orient='h', showmeans=True)
    
    results_summary = results.groupby('Model').mean()
    results_summary['std'] = results.groupby('Model').std()
    print(results_summary)
    
def test_embeddings_reproducibility(encoder, models, normalize=False,
                                    scoring='neg_mean_squared_error', cv=10):
    df = pd.read_csv('../Data/reproducibility.csv')
    df = df.sort_values(by='protein_id')
    x, y = df[['peptide', 'reproducibility']].values.T
    groups = LabelEncoder().fit_transform(df['protein_id'].values)
    dataset = PeptideDataset(x, y, aminoacids)
    loader = DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=Collate(),
    )
    embeds, targets = get_embeddings(encoder, loader)
    if normalize:
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    results = {'Model':[], 'Score':[]}
    for model_name, model in models:
        scores = cross_val_score(model, embeds, targets, groups=groups,
                                 cv=GroupKFold(),
                                 scoring=scoring, n_jobs=-1)
        results['Score'] += list(scores)
        results['Model'] += [model_name] * len(scores)
        
    results = pd.DataFrame(results)
    
    sns.boxplot(y='Model', x='Score', data=results,
                orient='h', showmeans=True)
    
    results_summary = results.groupby('Model').mean()
    results_summary['std'] = results.groupby('Model').std()
    print(results_summary)
    
def visualize_embeddings(embeds, targets, palette=None):
    ex, ey = TSNE(n_components=2).fit_transform(embeds).T
    sns.scatterplot(x=ex, y=ey, hue=targets.flatten(), palette=palette, alpha=0.7)

def get_loaders(x_train, y_train, x_test, y_test,
                Dataset, collate, batch_size):
    train_dataset = Dataset(x_train, y_train, aminoacids)
    test_dataset =  Dataset(x_test, y_test, aminoacids)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collate,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1024,
        shuffle=False,
        collate_fn=collate,
    )
    return train_loader, test_loader

def net_cross_val(model, x, y, loader_params, groups=None, 
                  kf=KFold(), epochs=20, early_stop=10):
    scores = []
    for train_index, test_index in kf.split(x, y, groups):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        train_loader, test_loader = get_loaders(x_train, y_train,
                                                x_test, y_test,
                                                **loader_params)
        curr_model = copy.deepcopy(model)
        history = curr_model.train(train_loader, epochs=epochs,
                                   early_stop=early_stop, verbose=False)
        scores.append(curr_model.evaluate(test_loader)['loss'])
    return np.array(scores)

def parallel_net_cross_val(model, x, y, loader_params, groups=None, 
                          kf=KFold(), epochs=20, early_stop=10):
    def get_score(train_index, test_index):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        train_loader, test_loader = get_loaders(x_train, y_train,
                                                x_test, y_test,
                                                **loader_params)
        curr_model = copy.deepcopy(model)
        history = curr_model.train(train_loader, epochs=epochs,
                                   early_stop=early_stop, verbose=False)
        return curr_model.evaluate(test_loader)['loss']
    
    scores = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(get_score)(train_index, test_index)
            for train_index, test_index in kf.split(x, y, groups)
        )
    return np.array(scores)

def ensemble_cross_val(model, x, y, loader_params, groups=None, 
                  kf=KFold(), epochs=20, early_stop=10):
    scores = []
    for train_index, test_index in kf.split(x, y, groups):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        train_loader, test_loader = get_loaders(x_train, y_train,
                                                x_test, y_test,
                                                **loader_params)
        curr_model = copy.deepcopy(model)
        curr_model.train(train_loader, test_loader=test_loader,
                         epochs=epochs, early_stop=early_stop,
                         verbose=False)
        preds, targets = curr_model.predict(test_loader)
        scores.append(sklearn.metrics.mean_squared_error(targets, preds))
    return np.array(scores)

def optimize(objective, space, mapping={}, evals=1):
    def map_params(params):
        mapped_params = {}
        for k, v in params.items():
            if k in mapping:
                mapped_params[k] = mapping[k](v)
            else:
                mapped_params[k] = v
        return mapped_params
        
    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=lambda x: objective(map_params(x)),
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=evals,
        trials=trials
    )
    
    history = defaultdict(list)
    for t in trials:
        history['loss'].append(t['result']['loss'])
        for k, v in t['misc']['vals'].items():
            if k in mapping:
                history[k].append(mapping[k](v[0]))
            else:
                history[k].append(v[0])
    
    return map_params(best), dict(history)