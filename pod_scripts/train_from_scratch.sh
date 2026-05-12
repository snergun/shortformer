python train.py \
    data/wikitext-103 \
    --task language_modeling \
    --arch transformer_lm_wiki103 \
    --max-update 1000 \
    --max-lr 1.0 \
    --t-mult 2 \
    --lr-period-updates 10 \
    --lr-scheduler cosine \
    --lr-shrink 0.75 \
    --warmup-updates 0 \
    --warmup-init-lr 1e-07 \
    --min-lr 1e-09 \
    --optimizer nag \
    --lr 0.0001 \
    --clip-norm 0.1 \
    --criterion adaptive_loss \
    --max-tokens 4608 \
    --update-freq 1 \
    --seed 1 \
    --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend no_c10d \
    --tokens-per-sample 128 \
    --required-batch-size-multiple 1 \
    --log-interval 5 \
    --keep-last-epochs 1 \
    --no-save \
    --temp-degree 1 \
    --temp-pieces 10 \
    --fixed-thresholds \
    # --plif-k 10 \
    # --plif-t 30 \
    # --plif-w-variance 1.0 \
    # --plif-lr 0.02 \



