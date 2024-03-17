---
license: apache-2.0
library_name: peft
tags:
- generated_from_trainer
metrics:
- bleu
base_model: Helsinki-NLP/opus-mt-ar-en
model-index:
- name: my_awesome___peft_model43
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_awesome___peft_model43

This model is a fine-tuned version of [Helsinki-NLP/opus-mt-ar-en](https://huggingface.co/Helsinki-NLP/opus-mt-ar-en) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2300
- Bleu: 44.8817
- Gen Len: 15.258

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 24
- eval_batch_size: 24
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Bleu    | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-------:|
| 1.3703        | 1.0   | 2292 | 1.2377          | 44.7837 | 15.2685 |
| 1.3613        | 2.0   | 4584 | 1.2312          | 44.8235 | 15.243  |
| 1.3753        | 3.0   | 6876 | 1.2300          | 44.8817 | 15.258  |


### Framework versions

- PEFT 0.8.2
- Transformers 4.37.2
- Pytorch 2.1.0+cu121
- Datasets 2.17.1
- Tokenizers 0.15.2