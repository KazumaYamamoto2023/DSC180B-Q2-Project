# trans2vec
This repo is a new implementation of trans2vec raised in the paper "[Who Are the Phishers? Phishing Scam Detection on Ethereum via Network Embedding](https://ieeexplore.ieee.org/document/9184813)"
. This version of the trans2vec may not perform as well as the paper because the original implementation and corresponding dataset are missing.

# Requirements:
- argparse
- scipy
- numpy
- pandas
- gensim
- sklearn
- numba

# Run the demo
```
python trans2vec.py
```

# Dataset
A part of original implementation can be found in directory "original", including spider codes and some transaction records crawled from [Etherscan](https://etherscan.io/).

If you want to learn more about the dataset used in the demo, please refer to the paper "[T-EDGE: Temporal WEighted MultiDiGraph Embedding for Ethereum Transaction Network Analysis](https://arxiv.org/abs/1905.08038)"


# Cite
Please cite our paper if you use this dataset in your own work:
```
@ARTICLE{wu2019t,
  TITLE={T-EDGE: Temporal WEighted MultiDiGraph Embedding for Ethereum Transaction Network Analysis},
  AUTHOR={Lin, Dan and Wu, Jiajing and Yuan, Qi and Zheng, Zibin},   
  JOURNAL={Frontiers in Physics},      
  VOLUME={8},      
  number={},
  PAGES={204},     
  YEAR={2020}
}
```

Please cite our paper if you use this code in your own work:
```
@ARTICLE{wu2019who,
  author={Wu, Jiajing and Yuan, Qi and Lin, Dan and You, Wei and Chen, Weili and Chen, Chuan and Zheng, Zibin},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems}, 
  title={Who Are the Phishers? Phishing Scam Detection on Ethereum via Network Embedding}, 
  year={2022},
  volume={52},
  number={2},
  pages={1156-1166},
}
```