PYTHON="/home/mengjian/anaconda3/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=cnn_mnist_b
batch_size=128

save_path="./save/${model}/${model}_lr${lr}_wd${wd}_eval/"
log_file="${model}_eval.log"
pretrianed_model="./save/cnn_mnist_b/cnn_mnist_b_lr5e-3_wd1e-4/model_best.pth.tar"

$PYTHON -W ignore train.py \
    --model ${model} \
    --save_path ${save_path} \
    --log_file ${log_file} \
    --depth 784 400 400 \
    --batch_size ${batch_size} \
    --resume ${pretrianed_model} \
    --evaluate;
