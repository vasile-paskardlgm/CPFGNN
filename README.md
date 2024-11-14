# Demo code implementation for the CPF (<u>C</u>oarsening-based <u>P</u>artition-wise <u>F</u>iltering)

## Environment
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
(dgl == 1.1.0)

## Executions
You can input the code below to reproduce partial of experiments.

**Cora:**
`python train.py --dataset cora --r_train 0.6 --r_val 0.2 --lr 0.01 --prop_lr 0.5 --weight_decay 0.0001 --prop_wd 0.0001 --dropout 0 --dprate 0.5 --alpha 0.2 --Init Mine_PPR --coarsening_ratio 0.1`

**Citeseer:**
`python train.py --dataset citeseer --r_train 0.6 --r_val 0.2 --lr 0.01 --prop_lr 0.5 --weight_decay 0.0005 --prop_wd 0.0001 --dropout 0.5 --dprate 0 --alpha 0.7 --Init Mine_PPR --coarsening_ratio 0.5`

**Pubmed:**
`python train.py --dataset pubmed --r_train 0.6 --r_val 0.2 --lr 0.01 --prop_lr 0.5 --weight_decay 0.0005 --prop_wd 0 --dropout 0.2 --dprate 0.2 --alpha 0.7 --Init Mine_PPR --coarsening_ratio 0.9`

**Roman-empire:**
`python train.py --dataset roman-empire --r_train 0.5 --r_val 0.25 --lr 0.01 --prop_lr 0.5 --weight_decay 0 --prop_wd 0 --dropout 0.2 --dprate 0 --alpha 0.7 --Init Mine --coarsening_ratio 0.9`

**Amazon-ratings**
`python train.py --dataset amazon-ratings --r_train 0.5 --r_val 0.25 --lr 0.1 --prop_lr 0.1 --weight_decay 0 --prop_wd 0 --dropout 0 --dprate 0.7 --alpha 0.7 --Init Mine_PPR --coarsening_ratio 0.7`

## Notes
- In our code implementation, CPF is structured as `CPFGNN`, since our method is in fact a GNN model.

- Please note that the code is only a demo code for confirming reproducibilty in conference submission. Full code will be supplemented after getting acceptance.

- Due to diversity of computational environments, the results will have (very slight) fluctuations. Using larger `runs` may alleviate the uncertainties. 

- The `--coarsening_ratio`, according to our paper, serves as a key hyperparameter optimized within {0.1,0.2,...,0.8,0.9}. Therefore, to minimize the influences from computational environments, we suggest refinetuning this hyperparameter when running our model.