# Reward Propagation using Graph Convolutional Networks
The repository contains the code for running the experiments in the paper [Reward Propagation using Graph Convolutional Networks](https://arxiv.org/abs/2010.02474) which was presented as a spotlight at NeurIPS 2020. The implementation is based on a few source codes: [gym-miniworld](https://github.com/maximecb/gym-miniworld), a good [pytorch PPO implementation](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) and Thomas Kipf's [pytorch GCN implementation](https://github.com/tkipf/pygcn).

# Installation

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Other requirements
pip install -r requirements.txt
pip install mujoco-py==2.0.2.2 #optional

#Installing PyGCN
python setup_gcn.py install
```

# Usage

## Atari

To launch a run on one of the Atari games, use the following command:
```bash
python control/main.py --num-frames 10000000 --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4  --gcn_alpha 0.9  --log-interval 1 --env-name ZaxxonNoFrameskip-v4 --seed 0 --entropy-coef 0.01  --use-logger --folder results
```


## MuJoCo

To launch a run on one of the delayed MuJoCo environments, use the following command:
```bash
python control/main.py --num-frames 3000000   --algo ppo --use-gae --lr 3e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --ppo-epoch 10 --num-steps 2048 --num-mini-batch 32 --gcn_alpha 0.6 --log-interval 1 --env-name Walker2d-v2 --seed 0 --entropy-coef 0.0  --use-logger --folder results --reward_freq 20
```

## Cite

If you found our paper useful or interesting, please cite it:

```bash
@misc{klissarov2020reward,
      title={Reward Propagation Using Graph Convolutional Networks}, 
      author={Martin Klissarov and Doina Precup},
      year={2020},
      eprint={2010.02474},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```