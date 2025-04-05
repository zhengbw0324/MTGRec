

IDS15=[9986,9987,9988,9989,9990,9991,9992,9993,9994,9995,9996,9997,9998,9999,10000]
IDS20=[9981,9982,9983,9984,9985,9986,9987,9988,9989,9990,9991,9992,9993,9994,9995,9996,9997,9998,9999,10000]
IDS25=[9976,9977,9978,9979,9980,9981,9982,9983,9984,9985,9986,9987,9988,9989,9990,9991,9992,9993,9994,9995,9996,9997,9998,9999,10000]
TAU_LIST=(1.0 3.0 5.0)



DATASET=Musical_Instruments
TOKEN=rqvae/sentence-t5-base_256,256,256,256

L=6
SATGE=[60,20,20,20,20,20,20]

for TAU in "${TAU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch \
        --main_process_port 12232 \
        --num_processes 4 pretrain.py \
        --dataset=$DATASET \
        --config_file=config/ptconfig.yaml \
        --token_prefix=$TOKEN \
        --lr=0.005 \
        --epochs=200 \
        --num_layers=$L \
        --num_decoder_layers=$L \
        --save_interval=10 \
        --val_delay=199 \
        --epoch_per_stage=$SATGE \
        --sem_id_epochs=$IDS25 \
        --tau=$TAU
done






DATASET=Industrial_and_Scientific
TOKEN=rqvae/sentence-t5-base_256,256,256,256

L=5
SATGE=[60,20,20,20,20,20,20]

for TAU in "${TAU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch \
        --main_process_port 12232 \
        --num_processes 4 pretrain.py \
        --dataset=$DATASET \
        --config_file=config/ptconfig.yaml \
        --token_prefix=$TOKEN \
        --lr=0.005 \
        --epochs=200 \
        --num_layers=$L \
        --num_decoder_layers=$L \
        --save_interval=10 \
        --val_delay=199 \
        --epoch_per_stage=$SATGE \
        --sem_id_epochs=$IDS15 \
        --tau=$TAU
done





DATASET=Video_Games
TOKEN=rqvae/sentence-t5-base_256,256,256,256

L=7
SATGE=[60,20,20,20,20,20,20]

for TAU in "${TAU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    accelerate launch \
        --main_process_port 12232 \
        --num_processes 4 pretrain.py \
        --dataset=$DATASET \
        --config_file=config/ptconfig.yaml \
        --token_prefix=$TOKEN \
        --lr=0.005 \
        --epochs=200 \
        --num_layers=$L \
        --num_decoder_layers=$L \
        --save_interval=10 \
        --val_delay=199 \
        --epoch_per_stage=$SATGE \
        --sem_id_epochs=$IDS25 \
        --tau=$TAU
done

