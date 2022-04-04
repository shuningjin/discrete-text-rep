## Discrete Text Representation


This is the codebase for our ACL 2020 paper:
[Discrete Latent Variable Representations for Low-Resource Text Classification](https://arxiv.org/abs/2006.06226) ([ACL portal](https://virtual.acl2020.org/paper_main.437.html), [video](https://slideslive.com/38929414/discrete-latent-variable-representations-for-lowresource-text-classification), [slides](slides/slides_jin2020discrete.pdf)).

```
@inproceedings{jin2020discrete,
    title = "Discrete Latent Variable Representations for Low-Resource Text Classification",
    author = "Shuning Jin and Sam Wiseman and Karl Stratos and Karen Livescu",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.acl-main.437"
}
```

<!-- we implement several discrete variational autoencoders (VAE) -->

**Table of Contents**
- [Software Environment](#1-Software-Environment)
- [Data](#2-Data)
- [Run Code](#3-Run-Code)


### 1 Software Environment
- packages are specified in [environment.yml](environment.yml)

- require conda3: [Anaconda 3](https://docs.anaconda.com/anaconda/install/) or [Miniconda 3](https://docs.conda.io/en/latest/miniconda.html)
    ```
    conda env create -f environment.yml
    conda activate discrete
    ```

### 2 Data

text classification datasets: AG News, DBPedia, Yelp Review Full

- from paper: [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification) (Zhang et al, NIPS 2015)

- [google drive link](https://drive.google.com/drive/u/3/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) shared by the author


- alternatively, you can get data by running the commands
  (specify /path/to/data/dir)
  ```bash
  # gdown is the library to download files from google drive
  # https://pypi.org/project/gdown/
  # already included in the envrionment.yml
  pip install gdown
  # download data from google drive
  bash scripts/download_data.sh /path/to/data/dir
  # random sample dev set 5000
  python scripts/train_dev_split.py /path/to/data/dir
  ```
- data directory should look like [this](scripts/data.md)

### 3 Run Code

- set environment variables
  ```bash
  export DISCRETE_DATA_DIR=/path/to/data/dir
  export DISCRETE_PROJECT_DIR=/path/to/project/dir
  ```

- TensorBoard (optional): to see [tensorboard](https://pytorch.org/docs/stable/tensorboard.html) output of pretraining
  ```bash
  # local server
  tensorboard --logdir [dir: tensorboard_train, tensorboard_val]
  open http://localhost:6006

  # remote server
  tensorboard --logdir [dir: tensorboard_train, tensorboard_val] --bind_all
  # replace remotehost name as prompted by the above command
  open http://remotehost:6006
  ```

- command examples:
  ```bash
  # Caveat: use SINGLE QUOTES in the commands
  # double quotes sometimes cause problems

  # to resume a previously interrupted pretraining, change `ckpt_path=none` to `ckpt_path=current`

  # vq pretrain
  python main.py \
  -c config/base.conf \
  -o 'expname=demo, runname=ag_sentence_vq,
  quantizer.level=sentence, quantizer.M=4, quantizer.K=256, quantizer.type=vq, vq.commitment_cost=1e-3, vq.use_ema=0,
  phase=pretrain, pretrain.use_noam=0, ckpt_path=none'

  # vq target train: 200 examples
  python main.py \
  -c config/base.conf \
  -o 'expname=demo, runname=ag_sentence_vq,
  quantizer.level=sentence, quantizer.M=4, quantizer.K=256, quantizer.type=vq, vq.commitment_cost=1e-3, vq.use_ema=0,
  phase=target_train, target=${target-tmpl}${target-200-tmpl}{test=0}, sub_runname=cls200, ckpt_path=current'

  # vq output pretrained encodings
  python main.py \
  -c config/base.conf \
  -o 'expname=demo, runname=ag_sentence_vq,
  quantizer.level=sentence, quantizer.M=4, quantizer.K=256, quantizer.type=vq, vq.commitment_cost=1e-3, vq.use_ema=0,
  phase=analyze, ckpt_path=current'

  # em pretrain
  python main.py -c config/base.conf \
  -o 'expname=demo, runname=ag_sentence_em,
  quantizer.level=sentence, quantizer.M=4, quantizer.K=256, quantizer.type=em,
  phase=pretrain, pretrain.em_iter=3, pretrain.use_noam=1, ckpt_path=none'

  # TODD: cat-vae, retrieval
  # more examples comming soon
  ```
- command explanation: [quick intro](https://github.com/nyu-mll/jiant/blob/master/tutorials/setup_tutorial.md#3-running-our-first-experiment), detailed intro comming soon.

### Acknowledgement
The coding logic is largely borrowed from and inspired by the [jiant](https://github.com/nyu-mll/jiant) library
```
@misc{wang2019jiant,
    author = {Alex Wang and Ian F. Tenney and Yada Pruksachatkun and Phil Yeres and Jason Phang and Haokun Liu and Phu Mon Htut and and Katherin Yu and Jan Hula and Patrick Xia and Raghu Pappagari and Shuning Jin and R. Thomas McCoy and Roma Patel and Yinghui Huang and Edouard Grave and Najoung Kim and Thibault F\'evry and Berlin Chen and Nikita Nangia and Anhad Mohananey and Katharina Kann and Shikha Bordia and Nicolas Patry and David Benton and Ellie Pavlick and Samuel R. Bowman},
    title = {\texttt{jiant} 1.3: A software toolkit for research on general-purpose text understanding models},
    howpublished = {\url{http://jiant.info/}},
    year = {2019}
}
```
