PYTHON="/home/mengjian/anaconda3/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=cnn_mnist
batch_size=1

save_path="./save/${model}/${model}_lr${lr}_wd${wd}_eval/"
log_file="${model}_eval.log"
pretrianed_model="./save/cnn_mnist/cnn_mnist_lr0.1_wd2e-4_p0.5/checkpoint.pth.tar"

$PYTHON -W ignore train.py \
    --model ${model} \
    --save_path ${save_path} \
    --log_file ${log_file} \
    --depth 784 400 400 \
    --batch_size ${batch_size} \
    --resume ${pretrianed_model} \
    --evaluate;
