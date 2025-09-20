# Compare Huggingface model vs Fairseq
## exp1 > (w/o cp weights + w/o emb_loss)
- 1_fairseq: faiseq model
- 1_hf: huggingface model
- 1_hf_v2 (huggingface model + 100% dataset)
## exp2 > (cp weights + w/o emb_loss)
- 2_fairseq
## exp3 > (w/o cp weights)
- 3_hf (student project MLP w/o recon)
- 3_hf_v2 (teacher project ShallowAutoencoder)