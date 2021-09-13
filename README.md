# MSc-Thesis
##  Improving Reliability of Shotgun Proteomics with Machine Learning

<!-- <p align="justify"> This thesis aims to improve the reliability of shotgun proteomics. The large scale study of proteins suffers from yet not fully understood problems in accurate measurement of proteins. The thesis addresses the peptide detectability prediction problem and investigates the reasons for the poor reproducibility of proteomics experiments. It includes the background on deep learning for sequential modelling and a literature review of successful machine learning methodologies for proteomics data. Here we present an end-to-end ensemble of recurrent neural networks that can predict peptide detectability with high accuracy based on amino acid sequences. Our model outperforms other state-of-the-art methods on Homo sapiens and Mus musculus benchmark data sets. We find that the model generalizes well across different species. Moreover, we show there is no statistical evidence supporting the hypothesis that the peptide distance from the protein centre impacts the reproducibility of shotgun proteomics experiments. The thesis also includes the comparison of different peptide encoding methods and connects peptide retention time to the reproducibility of mass spectrometry results. </p> -->

### Data
Contains data sets used in this thesis

### deep
Custom package for deep neural networks based on PyTorch
* packages - contains the relevant imports
* utils - helper functions
* datasets - implementations of data preprocessors and loaders
* losses - custom losses
* models - implementations of neural networks and training procedures

### Detectability
Folder with code for running peptide detectability prediction experiments
* Results - folder with saved experiment results
* DetectabilityHyperopt - notebook for running hyperparameter search
* DetectabilityExperiments - notebook for running experiments
* DetectabilityResults - a notebook with analyses of the results

### Reproducibility
Folder with code for running peptide reproducibility prediction experiments
* Results - folder with saved experiment results
* ProteinCentreHypothesis - notebook for testing the protein centre hypothesis
* BioFeatures - notebook for creating other biological peptide features
* ContextualEmbeddings - notebook used for inferring the peptide embeddings from pre-trained models
* Experiments - a notebook with experiments
* DeepExperiments - a notebook with deep learning experiments
* ReproducibilityResutls - a notebook with analyses of the results
