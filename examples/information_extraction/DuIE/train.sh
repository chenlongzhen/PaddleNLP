set -eux

export device=gpu
export BATCH_SIZE=64
export LR=2e-5
export EPOCH=12

if [ $# -ne 0 ]; then
  device=$1
python  run_duie.py \
                            --device $device \
                            --seed 42 \
                            --do_train \
                            --data_path ./data \
                            --max_seq_length 128 \
                            --batch_size $BATCH_SIZE \
                            --num_train_epochs $EPOCH \
                            --learning_rate $LR \
                            --warmup_ratio 0.06 \
                            --output_dir ./checkpoints \
                            --predict_data_file ./data/duie_test2.json \
                            --save_steps 10000 \
                            --pretrian_model_name ernie-3.0-base-zh
else
  echo "device gpu"
  unset CUDA_VISIBLE_DEVICES
  python -u -m paddle.distributed.launch --gpus "0" run_duie.py \
                              --device $device \
                              --seed 42 \
                              --do_train \
                              --data_path ./data \
                              --max_seq_length 128 \
                              --batch_size $BATCH_SIZE \
                              --num_train_epochs $EPOCH \
                              --learning_rate $LR \
                              --warmup_ratio 0.06 \
                              --output_dir ./checkpoints \
                              --predict_data_file ./data/duie_test2.json \
                              --save_steps 10000 \
                              --pretrian_model_name ernie-3.0-base-zh
fi
shutdown
