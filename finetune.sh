

DATASET=Musical_Instruments
TOKEN=rqvae/sentence-t5-base_256,256,256,256

L=6
CKPT=./ckpt/${DATASET}/XXXX/XXXX
CKPT_EPOCHS=(90 100 110 120)
SEM_IDS=(9985,9986,9987,9988,9989,9990,9991,9992)


for CKPT_EPOCH in "${CKPT_EPOCHS[@]}"; do
for SEM_ID in "${SEM_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    accelerate launch \
        --main_process_port 25232 \
        --num_processes 4 finetune.py \
        --dataset=${DATASET} \
        --config_file=config/ftconfig.yaml \
        --token_prefix=${TOKEN} \
        --lr=0.0002 \
        --warmup_steps=0 \
        --num_layers=$L \
        --num_decoder_layers=$L \
        --epochs=100 \
        --patience=10 \
        --pretrained_model=${CKPT}_${CKPT_EPOCH}.pth \
        --sem_id_epochs=[$SEM_ID]
done
done





DATASET=Industrial_and_Scientific
TOKEN=rqvae/sentence-t5-base_256,256,256,256

L=5
CKPT=./ckpt/${DATASET}/XXXX/XXXX
CKPT_EPOCHS=(90 100 110 120)
SEM_IDS=(9990,9991,9992,9993,9994,9995,9996)


for CKPT_EPOCH in "${CKPT_EPOCHS[@]}"; do
for SEM_ID in "${SEM_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    accelerate launch \
        --main_process_port 25232 \
        --num_processes 4 finetune.py \
        --dataset=${DATASET} \
        --config_file=config/ftconfig.yaml \
        --token_prefix=${TOKEN} \
        --lr=0.0002 \
        --warmup_steps=0 \
        --num_layers=$L \
        --num_decoder_layers=$L \
        --epochs=100 \
        --patience=10 \
        --pretrained_model=${CKPT}_${CKPT_EPOCH}.pth \
        --sem_id_epochs=[$SEM_ID]
done
done





DATASET=Video_Games
TOKEN=rqvae/sentence-t5-base_256,256,256,256

L=7
CKPT=./ckpt/${DATASET}/XXXX/XXXX
CKPT_EPOCHS=(120 130 140 150)
SEM_IDS=(9985,9986,9987,9988,9989,9990,9991,9992)


for CKPT_EPOCH in "${CKPT_EPOCHS[@]}"; do
for SEM_ID in "${SEM_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    accelerate launch \
        --main_process_port 25232 \
        --num_processes 4 finetune.py \
        --dataset=${DATASET} \
        --config_file=config/ftconfig.yaml \
        --token_prefix=${TOKEN} \
        --lr=0.0002 \
        --warmup_steps=0 \
        --num_layers=$L \
        --num_decoder_layers=$L \
        --epochs=100 \
        --patience=10 \
        --pretrained_model=${CKPT}_${CKPT_EPOCH}.pth \
        --sem_id_epochs=[$SEM_ID]
done
done
