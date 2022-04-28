# Personalized Federated Learning with Graph

This is the original pytorch implementation of Structural Federated Learning(SFL) in the following paper

[Personalized Federated Learning with Graph, IJCAI 2022] (https://arxiv.org/abs/2203.00829).

<p align="center">
  <img width="" height="400" src=./fig/overall.png>
</p>

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt
## Data Preparation

### Traffic datasets
Download the METR-LA and PEMS-BAY dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git) . Move them into the data folder. 

```

# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```

## Demo

SFL
python main.py --dataset cifar10 --com_round 20  --shards 5 --agg graph

SFL*
python main.py --dataset cifar10 --com_round 20  --shards 5 --agg graph_v3
