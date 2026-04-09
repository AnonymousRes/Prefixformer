#! /bin/bash


DATA=/workspace/lra/aan
SAVE_ROOT=/workspace/lra_saved/aan
exp_name=aan_prefixformer
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/DWSSMHP25/out_log
#rm -rf ${SAVE}
#rm -f ${SAVE_LOG}/${exp_name}_log.txt
#mkdir -p ${SAVE}

model=prefixformer_lra_aan
export CUDA_VISIBLE_DEVICES=1

#python -u /workspace/DWSSMHP25/train.py ${DATA} \
#    --seed 2026 \
#    --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
#    -a ${model} --task lra-text --input-type text \
#    --encoder-layers 6 --encoder-attention-heads 8 --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
#    --activation-fn 'silu' --attention-activation-fn 'softmax' \
#    --norm-type 'scalenorm' --sen-rep-type 'mp' \
#    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#    --optimizer adam --lr 0.000866 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
#    --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.0368 \
#    --batch-size 32 --max-epoch 100 --sentence-avg --update-freq 1 \
#    --lr-scheduler 'fixed' \
#    --keep-last-epochs 1 --max-sentences-valid 32 \
#    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt
#
python /workspace/DWSSMHP25/fairseq_cli/validate.py ${DATA} --task lra-text --batch-size 32 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
#python /workspace/DWSSMHP25/fairseq_cli/validate.py ${DATA} --task lra-text --batch-size 32 --valid-subset test --path ${SAVE}/checkpoint_best.pt



