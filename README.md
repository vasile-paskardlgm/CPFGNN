# Demo code implementation for the CPF (<u>C</u>oarsening-based <u>P</u>artition-wise <u>F</u>iltering)

## Environmental details
Ubunto-22.04  
CUDA-11.8  
python == 3.10  
torch == 2.4.0  
torchvision == 0.19.1  
torchaudio == 2.4.1  
torch-geometric == 2.6.0  
pygsp == 0.5.1  
pyg_lib == 0.4.0  
torch_cluster == 1.6.3  
torch_scatter == 2.1.2  
torch_sparse == 0.6.18  
torch_spline_conv == 1.2.2  
sortedcontainers == 2.4.0  
fast-pytorch-kmeans == 0.2.2  

## Executions
You can execute the commands below to reproduce partial of the experiments.

**Cora:**
`python train.py --dataset cora --finetuned --gpu`

**Citeseer:**
`python train.py --dataset citeseer --finetuned --gpu`

**Pubmed:**
`python train.py --dataset pubmed --finetuned --gpu`

**Roman-empire:**
`python train.py --dataset roman-empire --finetuned --gpu`

**Amazon-ratings**
`python train.py --dataset amazon-ratings --finetuned --gpu`

## Notes
- This code is intended solely for demonstration purposes to confirm reproducibility for the conference submission. Full access to the code will be made available upon acceptance of the paper.

- The CPF method in our implementation is structured within the `CPFGNN` model class, as it is fundamentally a GNN-based approach.

- Due to differences in computational setups, the results may experience minor fluctuations when compared to the ones presented in the paper. Running the code multiple times can help mitigate these discrepancies.