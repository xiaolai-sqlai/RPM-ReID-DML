# RPM
The official repository for Revisiting Part Model: From Person Re-identification to Deep Metric Learning.

## Prepare Datasets
Download the person datasets, vehicle datasets, and fine-grained Visual Categorization/Retrieval datasets.

Then unzip them and rename them under your "dataset_root" directory like
```bash
dataset_root
├── Market-1501-v15.09.15
├── DukeMTMC-reID
├── MSMT17
├── cuhk03-np
├── VeRi
├── VehicleID_V1.0
├── CARS
├── CUB_200_2011
├── Stanford_Online_Products
├── In-shop
└── University-Release
```

## Training
We prepared the ImageNet Pretrained RegNet backbone in "./pretrain".

### Train on Market1501
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset market1501 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.961589 top5:0.986936 top10:0.991390 mAP:0.904650
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset market1501 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.964074 top5:0.986342 top10:0.992577 mAP:0.908825

### Train on DukeMTMC
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.915619 top5:0.956912 top10:0.969031 mAP:0.824559
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.918312 top5:0.960054 top10:0.971275 mAP:0.827411

### Train on CUHK03 Detected
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 384 --img-width 128 --batch-size 48 --lr 1.0e-1 --dataset npdetected --gpus 0 --epochs 5,155 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.842143 top5:0.931429 top10:0.963571 mAP:0.811973
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 384 --img-width 128 --batch-size 48 --lr 1.0e-1 --dataset npdetected --gpus 0 --epochs 5,155 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.850714 top5:0.934286 top10:0.957143 mAP:0.814122

### Train on CUHK03 Labeled
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 384 --img-width 128 --batch-size 48 --lr 1.0e-1 --dataset nplabeled --gpus 0 --epochs 5,155 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.874286 top5:0.954286 top10:0.975000 mAP:0.844478
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 384 --img-width 128 --batch-size 48 --lr 1.0e-1 --dataset nplabeled --gpus 0 --epochs 5,155 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.871429 top5:0.947857 top10:0.973571 mAP:0.841431

### Train on MSMT17
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 384 --img-width 128 --batch-size 48 --lr 1.0e-1 --dataset msmt17 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.863024 top5:0.930697 top10:0.947080 mAP:0.672814
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 384 --img-width 128 --batch-size 48 --lr 1.0e-1 --dataset msmt17 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.866455 top5:0.930268 top10:0.947680 mAP:0.676779

### Train on VeRI776
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 256 --img-width 256 --batch-size 48 --lr 1.0e-1 --dataset veri776 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.973778 top5:0.985697 top10:0.993445 mAP:0.830682
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 256 --img-width 256 --batch-size 48 --lr 1.0e-1 --dataset veri776 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.976758 top5:0.989273 top10:0.991657 mAP:0.828620

### Train on VehicleID
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 256 --img-width 256 --batch-size 256 --lr 2.0e-1 --dataset vehicleid --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.892324 top5:0.981732 top10:0.992447 mAP:0.911746
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 256 --img-width 256 --batch-size 256 --lr 2.0e-1 --dataset vehicleid --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.3 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
top1:0.884244 top5:0.983664 top10:0.992798 mAP:0.905112

### Train on Car196
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 224 --img-width 224 --batch-size 48 --lr 1.0e-1 --dataset car196 --gpus 0 --epochs 5,45 --instance-num 6 --erasing 0.1 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.914894 Recall@2:0.949945 Recall@4:0.968885 Recall@8:0.981675 NMI:0.793197
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 224 --img-width 224 --batch-size 48 --lr 1.0e-1 --dataset car196 --gpus 0 --epochs 5,45 --instance-num 6 --erasing 0.1 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.907023 Recall@2:0.941090 Recall@4:0.961628 Recall@8:0.974665 NMI:0.775495

### Train on CUB200
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 224 --img-width 224 --batch-size 48 --lr 2.0e-3 --dataset cub200 --gpus 0 --epochs 5,45 --instance-num 6 --erasing 0.2 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.718433 Recall@2:0.813639 Recall@4:0.877954 Recall@8:0.923869 NMI:0.732626
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 224 --img-width 224 --batch-size 48 --lr 2.0e-3 --dataset cub200 --gpus 0 --epochs 5,45 --instance-num 6 --erasing 0.1 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.710500 Recall@2:0.806381 Recall@4:0.875422 Recall@8:0.929946 NMI:0.726015

### Train on SOP
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset sop --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.1 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.843542 Recall@10:0.938283 NMI:0.919845
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 224 --img-width 224 --batch-size 128 --lr 1.0e-1 --dataset sop --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.1 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.844765 Recall@10:0.938002 NMI:0.920118

### Train on DeepFashion
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 224 --img-width 224 --batch-size 128 --lr 2.0e-1 --dataset deepfashion --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.1 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.945054 Recall@10:0.988184 Recall@20:0.992123 Recall@30:0.993600 Recall@40:0.994373 NMI:0.928027
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 224 --img-width 224 --batch-size 128 --lr 2.0e-1 --dataset deepfashion --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.1 --num-part 2 --num-stripe 1 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 25
```
Recall@1:0.945914 Recall@10:0.988043 Recall@20:0.991560 Recall@30:0.993459 Recall@40:0.994795 NMI:0.928868

### Train on University1652
```bash
python train.py --net regnet_y_1_6gf --decoder rpm_max --img-height 224 --img-width 224 --batch-size 24 --lr 5.0e-2 --dataset university1652 --gpus 0 --epochs 3,15 --instance-num 6 --erasing 0.1 --num-part 2 --num-stripe 0 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 15
```
top1:0.922967 top5:0.950071 top10:0.952924 mAP:0.900724<br>
top1:0.897081 top5:0.964364 top10:0.973742 mAP:0.912387
```bash
python train.py --net regnet_y_1_6gf_prelu --decoder rpm_max --img-height 224 --img-width 224 --batch-size 24 --lr 5.0e-2 --dataset university1652 --gpus 0 --epochs 3,15 --instance-num 6 --erasing 0.1 --num-part 2 --num-stripe 0 --use-global True --triplet-weight 1.0 --feat-num 256 --ada-gamma 0.10 --freeze stem --dataset-root ../datasets --ema-ratio 0.80 --ema-extra 15
```
top1:0.927247 top5:0.950071 top10:0.955777 mAP:0.901062<br>
top1:0.897715 top5:0.966108 top10:0.975802 mAP:0.913270

## Contact
If you have any questions, please contact us by email(laishenqi@qq.com).