#! /bin/bash


DATA=/workspace/lra/imdb-4000
SAVE_ROOT=/workspace/lra_saved/imdb
exp_name=imdb_prefixformer
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/DWSSMHP25/out_log
#rm -rf ${SAVE}
#rm -f ${SAVE_LOG}/${exp_name}_log.txt
#mkdir -p ${SAVE}

model=prefixformer_lra_imdb
export CUDA_VISIBLE_DEVICES=1

#python -u /workspace/DWSSMHP25/train.py ${DATA} \
#    --seed 2026 \
#    --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
#    -a ${model} --task lra-text --input-type text \
#    --k-times 1 \
#    --encoder-layers 4 --encoder-attention-heads 8 --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
#    --activation-fn 'silu' --attention-activation-fn 'softmax' \
#    --norm-type 'scalenorm' --sen-rep-type 'mp' \
#    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#    --optimizer adam --lr 0.006 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1 --clip-mode 'total' \
#    --dropout 0.08 --weight-decay 0.01 \
#    --batch-size 25 --sentence-avg --update-freq 2 --required-batch-size-multiple 1 \
#    --lr-scheduler linear_decay --total-num-update 50000 --max-update 50000 --warmup-updates 10000 --warmup-init-lr '1e-07' --end-learning-rate 0.0\
#    --keep-last-epochs 1 \
#    --max-sentences-valid 25 \
#    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt

python /workspace/DWSSMHP25/fairseq_cli/validate.py ${DATA} --task lra-text --batch-size 64 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
#--lr-scheduler='linear_decay' --end-learning-rate 0.0 \
#--lr-scheduler linear_decay --total-num-update 60000 --end-learning-rate 0.0
#--warmup-updates 5000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --max-sentences-valid 64
#--patience 20
#--lr-scheduler='linear_decay' --end-learning-rate 0.0 --warmup-updates 1000 --warmup-init-lr '1e-07' \