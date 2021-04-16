PYTHON="/home/mengjian/anaconda3/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=cnn_mnist_q
epochs=30
batch_size=128

wd=1e-4
lr=5e-3

save_path="./save/${model}/${model}_lr${lr}_wd${wd}/"
log_file="${model}_lr${lr}_wd${wd}.log"

$PYTHON -W ignore train.py \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --schedule 20 25 \
    --gammas 0.1 0.1 \
    --depth 784 400 400 \
    --batch_size ${batch_size} \
    --weight_decay ${wd} \
    --binary;
