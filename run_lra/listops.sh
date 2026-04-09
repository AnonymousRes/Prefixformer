#! /bin/bash


DATA=/workspace/lra/listops
SAVE_ROOT=/workspace/lra_saved/listops
exp_name=listops_prefixformer
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/DWSSMHP25/out_log
#rm -rf ${SAVE}
#rm -f ${SAVE_LOG}/${exp_name}_log.txt
#mkdir -p ${SAVE}

model=prefixformer_lra_listop
export CUDA_VISIBLE_DEVICES=1

#python -u /workspace/DWSSMHP25/train.py ${DATA} \
#    --seed 2026 \
#    --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
#    -a ${model} --task lra-text --input-type text \
#    --encoder-layers 6 --encoder-attention-heads 8 --encoder-embed-dim 160 --encoder-ffn-embed-dim 160 \
#    --activation-fn 'silu' --attention-activation-fn 'softmax' \
#    --norm-type 'rmsnorm' --sen-rep-type 'cls' \
#    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#    --optimizer adam --lr 0.0005 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
#    --dropout 0.2 --attention-dropout 0.0 --act-dropout 0.2 --weight-decay 0.1666 \
#    --batch-size 64 --max-epoch 150 --sentence-avg --update-freq 1 --max-sentences-valid 256 \
#    --lr-scheduler fixed \
#    --keep-last-epochs 1 --required-batch-size-multiple 1 \
#    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt

python /workspace/DWSSMHP25/fairseq_cli/validate.py ${DATA} --task lra-text --batch-size 64 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
#    --lr-scheduler fixed --total-num-update 90000 --end-learning-rate 0.0 --warmup-updates 3000 --warmup-init-lr '1e-07' \