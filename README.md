## NumerSense: NumerSense: Probing Numerical Commonsense Knowledge of BERTs

Code & Data for EMNLP 2020 paper:

```bibtex
@inproceedings{lin2020numersense,
  title={Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-trained Language Models},
  author={Bill Yuchen Lin and Seyeon Lee and Rahul Khanna and Xiang Ren}, 
  booktitle={Proceedings of EMNLP},
  year={2020},
  note={to appear}
}
```

### Installation 

```bash
conda create -n numersense python=3.7
conda activate numersense
# install torch seperately at https://pytorch.org/get-started/locally/ if needed
pip install happytransformer
pip install transformers==3.0.2
pip install tensorboardX
mkdir pred_results

# Optional:
# Install apex following https://github.com/NVIDIA/apex#linux
```

### Probing Experiments 

```bash
python src/mlm_infer.py bert-base data/test.core.masked.tsv
```

```bash
python src/gpt_infer.py gpt2 data/test.core.masked.tsv
```

#### Fine-tune a MLM model 
```bash
mkdir saved_models
CUDA_VISIBLE_DEVICES=0 python src/finetune_mlm.py \
  --output_dir=saved_models/finetuned_bert_base --overwrite_output_dir \
  --model_type=bert \
  --model_name_or_path=bert-base-uncased \
  --do_train \
  --train_data_file=data/gkb_best_filtered.txt  \
  --do_eval \
  --eval_data_file=data/wiki_complete.txt \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --block_size 64 \
  --logging_steps 100 \
  --num_train_epochs 5 \
  --line_by_line --mlm \
  --fp16
```

### Dataset

