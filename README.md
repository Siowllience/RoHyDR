
This is the official implementation of the paper '[RoHyDR: Robust Hybrid Diffusion Recovery for Incomplete Multimodal Emotion Recognition](http://arxiv.org/abs/2505.17501)' 

## Quick Start

### Environments

Please refer to `requirements.txt` for more information.


### Datasets
You can access the CMU-MOSI and CMU-MOSEI datasets via the CMU Multimodal Data SDK or obtain them from previously released open-source multimodal emotion recognition projects. We also provided the datasets in links bellow: 
- [MOSI](https://pan.baidu.com/s/1DEqb_K79jkk6eUdX4NMYMQ?pwd=6rpv)
- [MOSEI](https://pan.baidu.com/s/1VcsE_lZU265x6roJXbcT_w?pwd=wigp)

Downloaded datasets can be put into the `dataset/` directory and organized as `dataset/MOSI/aligned_50.pkl` and `dataset/MOSEI/aligned_50.pkl`. Alternatively, you can modify the dataset path in `config/config.json`.



### Run the Code

```
python train.py
```

### Reproduce
To reproduce our results, please download the pretrained weights [here]() and put them in the `pretrained/` folder. 


## Tips

- You can refer to `trains/mainNets/model/rohydr.py` for the proposed Hybrid Recovery Method.
- The Multi-Stage Optimization Strategy can be found in `trains/mainNets/ROHYDR.py`.
- The training settings used in our experiments are available in `config/config.json`.
