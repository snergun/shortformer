# fairseq-eval-lm data/wikitext-103 \
#     --path checkpoint_best.pt \
#     --sample-break-mode none \
#     --gen-subset valid  \
#     --max-sentences 1 \
#     --sliding-inf 1 \
#     --context-window 511 \
#     --max-tokens 512
fairseq-eval-lm data/wikitext-103 \
    --path checkpoints/shortformer/model.pt \
    --sample-break-mode none \
    --gen-subset valid  \
    --max-sentences 1 \
    --results-path checkpoints/shortformer/results/ \
    --save-layers -1 \
    --save-probs
fairseq-eval-lm data/wikitext-103 \
    --path checkpoints/shortformer/model.pt \
    --sample-break-mode none \
    --gen-subset test  \
    --max-sentences 1 \
    --results-path checkpoints/shortformer/results/ \
    --save-layers -1 \
    --save-probs
