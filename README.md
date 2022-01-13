
# My VRDL HW4

Use HAN to do super resolution

## My environment
GUP: GTX3060

CUDA: 11.3
cudnn: 8.2.1

## Training

To train the model(s) in the paper, run this command:

```train
cd src
python main.py --template HAN --save HANx3 --scale 3 --reset --save_results --patch_size 60
```

## my awesome Models

You can download my models here:

- [My awesome model](https://drive.google.com/file/d/1SW80OPsuYtOQK41kYd079Nc98k7L-nXb/view?usp=sharing) 


## Inference
to reproduce submission file

```Inference
cd src
python eval.py --template HAN --scale 2 --pre_train path/to/model
```



