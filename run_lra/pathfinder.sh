#! /bin/bash


DATA=/workspace/lra/pathfinder
SAVE_ROOT=/workspace/lra_saved/pathfinder
exp_name=pathfinder_prefixformer
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/DWSSMHP25/out_log
#rm -rf ${SAVE}
#rm -f ${SAVE_LOG}/${exp_name}_log.txt
#mkdir -p ${SAVE}

model=prefixformer_lra_pf32
export CUDA_VISIBLE_DEVICES=1

#python -u /workspace/DWSSMHP25/train.py ${DATA} \
#    --seed 2026 \
#    --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
#    -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
#    --encoder-layers 6 --encoder-attention-heads 8 --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
#    --activation-fn 'silu' --attention-activation-fn 'softmax' \
#    --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
#    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-mode='total' --clip-norm 1.0 \
#    --dropout 0.10 --weight-decay 0.15 \
#    --batch-size 128 --sentence-avg --update-freq 1 \
#    --lr-scheduler 'linear_decay' --total-num-update 250000 --max-update 250000 --end-learning-rate 0.0 --warmup-updates 50000 --warmup-init-lr '1e-07' \
#    --keep-last-epochs 1 --max-sentences-valid 150 \
#    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt

#        --lr-scheduler 'fixed' --max-epoch 100 \
#--lr-scheduler 'linear_decay' --total-num-update 250000 --end-learning-rate 0.0 --warmup-updates 50000 --warmup-init-lr '1e-07' \

python /workspace/DWSSMHP25/fairseq_cli/validate.py ${DATA} --task lra-image --batch-size 64 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt