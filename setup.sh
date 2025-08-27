# needs python 3.10

sudo apt update -y
sudo apt install python3.10 python3.10-venv python3.10-dev -y
python3.10 -m venv .venv

.venv/bin/python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

.venv/bin/python -m pip install wheel setuptools

.venv/bin/python -m pip install -r requirements.txt


#huggingface-cli download liuhaotian/llava-v1.5-13b --local-dir llava-v1.5-13b --local-dir-use-symlinks False
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors

curl -L -o weights.zip https://www.kaggle.com/api/v1/datasets/download/bpwqsdd/sdbsettjstjzk
unzip weights.zip
rm weights.zip

.venv/bin/python -m pip install jupyterlab ipykernel
.venv/bin/python -m ipykernel install --user --name=virtualenv --display-name "Python (supir)"

