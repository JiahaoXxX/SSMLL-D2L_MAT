device_id=0
dataset_dir='./data'

for lb_ratio in 0.01 0.05 0.1 0.15 0.2
do
    for dataset_name in 'voc' 'coco' 'nus'
    do
    CUDA_VISIBLE_DEVICES=$device_id python run_warmup.py \
    --dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
    --net resnet50 --loss_lb asl --warmup_epochs 12 --lr 1e-4

    CUDA_VISIBLE_DEVICES=$device_id python run_main.py \
    --dataset_name $dataset_name --dataset_dir $dataset_dir --lb_ratio $lb_ratio \
    --net resnet50 --loss_lb asl --loss_ub asl --warmup_epochs 12 --lr 1e-4 \
    --early_stop --cat_strategy fB --beta 0.5  
    done
done