# Unified-Width Adaptive Dynamic Network for All-In-One Image Restoration

PyTorch implementation for Unified-Width Adaptive Dynamic Network for All-In-One Image Restoration **(U-WADN)**.


<img src=".\Figure\Fig2.png"/>

## Dependencies

* Python == 3.8.11
* Pytorch == 1.10.0 
* mmcv-full == 2.0.0

## Dataset

You could find the dataset we used in the paper at following:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://ece.uwaterloo.ca/~k29ma/exploration/), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (OTS)

## Testing

The pretrained model is upload in ckpt/best_ckpt/best.pth. To test with the pretrained model, please:

```bash
python test.py --mode 3
```
If you only want  to test one of these tasks, please specific the test mode as 0, 1 or 2. (0 for denoising, 1 for deraining and 2 for dehazing).

## Training

If you want to re-train our model, you need to first put the training set into the data/. As the proposed U-WADN has 2 training 
steps as 1). Training of WAB and 2). Training of WS.

The training of WAB can be implemented by
```bash
python train.py --stage 1
```

The training of WS can be implemented by
```bash
python train_selector.py --stage 2
```


## Acknowledgement

This repo is built upon the framework of [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet/tree/main/data), and we borrow some code from [Slimmable Network](https://github.com/JiahuiYu/slimmable_networks), thanks for their excellent work!

