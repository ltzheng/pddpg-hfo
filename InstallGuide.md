# Environment Configuration for Multi-Agent RL in RoboCup 2D HFO

## Configure Sources

```bash
sudo nano /etc/apt/sources.list
```

[TUNA source](https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/)

## Install Anaconda

[TUNA source](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

Download and install:

```bash
bash Anaconda...
```

Add environment variable:

```bash
source ~/.bashrc
```

Test `pip`/`conda`:

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

## Create Virtual Environment

```bash
conda create --name marl python=3.8
conda activate marl
```

```bash
conda deactivate ...
conda remove --name ...
```

## Upgrade Pip

```bash
pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple pip -U
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
```

## Install Pytorch

[Official website](https://pytorch.org/get-started/locally/)

None CUDA version:
```bash
conda install pytorch torchvision torchaudio cpuonly pytorch
```

CUDA 11.0 version:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.0 pytorch
```

Validation:

```bash
python
```
```python
>>> import torch
>>> torch.cuda.is_available()
```

## Install Gym

[Official website](https://gym.openai.com/docs/#installation)

```bash
pip install gym
```

## Install Gym-soccer (RoboCup 2D HFO)

Install Qt4:

```bash
sudo apt-get install qt4-qmake libqt4-dev
```

```bash
git clone https://github.com/openai/gym-soccer.git
cd gym-soccer
pip install -e .
```


## Install Ray RLlib

```bash
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install cmake
sudo apt-get install -y cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev libboost-all-dev libsdl2-dev swig
pip install ray[rllib]
```

## Install Redis

```bash
sudo apt install redis-server
```