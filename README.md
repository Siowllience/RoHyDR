
This is the official implementation of the paper '[RoHyDR: Robust Hybrid Diffusion Recovery for Incomplete Multimodal Emotion Recognition](http://arxiv.org/abs/2505.17501)' 

## Quick Start

### Environments

Please refer to `requirements.txt` for more information.


### Datasets
We provided the datasets [here](https://pan.baidu.com/s/1LoxKH5U60BQJ2O7OGIKGTw?pwd=gvfi). Downloaded datasets can be put into the `dataset/` directory. Alternatively, you can modify the dataset path in `config/config.json`.



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
