### Install UV
https://docs.astral.sh/uv/getting-started/installation/ 
참고하여 설치합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

python 3.12 가 필요합니다.
```bash
uv python install 3.12
```

### Model Repository 
모델은 huggingface repository 에 저장하였습니다.
[SFT-Model](https://huggingface.co/kwkim1030/Qwen3-0.6-Countdwon-SoS-SFT)
https://huggingface.co/kwkim1030/Qwen3-0.6-Countdwon-SoS-SFT
[RLHF-Model](https://huggingface.co/kwkim1030/Qwen3-0.6-Countdwon-SoS-RLOO)
https://huggingface.co/kwkim1030/Qwen3-0.6-Countdwon-SoS-RLOO

### Git Clone
#### 데이터셋 구성
데이터셋 구성을 위해 Stream of Search 를 클론합니다.
```bash
git clone https://github.com/kanishkg/stream-of-search.git
```
해당 저장소의 세팅 방법을 따라합니다.
1. Install conda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
2. Create a conda environment
```bash
conda create -n sos python=3.11
conda activate sos
```
3. Install the required packages
```bash
pip install -r requirements.txt
```
데이터셋 생성
```bash
sh scripts/gen_task.sh
```

#### 훈련
```bash
git clone https://github.com/kwkim1030/rl-project.git
```

```bash
cd rl-project
uv venv
conda deactivate
source .venv/bin/activate 
uv sync
```
Stream of Search 의 데이터를 복사합니다.
```bash
mkdir data
cp  ../stream-of-search/data/b4_3_random/train1_b4_t100_n500000_random.json ./data/train1_b4_t100_n500000_random.json
```

### sft model train
```bash
uv run accelerate launch src/rl_project/train_sft.py
```

### rlhf model train
```bash
uv run accelerate launch src/rl_project/train_rloo.py
```