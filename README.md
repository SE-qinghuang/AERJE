# AERJE

- Code for [``API Entity and Relation Joint Extraction From Text via Dynamic Prompt-tuned Languange Model``](https://dl.acm.org/doi/10.1145/3607188)
- Please contact ybsun@jxnu.edu.cn for questions and suggestions.

## Update
- [2023-07-08] Update main code.

## Requirements
- Python (verified on 3.8)
- CUDA (verified on 11.1/10.2)

## Quick Start

### Download Dataset
[AERJE_Dataset](https://drive.google.com/file/d/1X6pQQhIspNHj2y6GlJaNW0bt3VaQjkXe/view)

### Download Pretrained Models
Dynamic Prompt Generator ([BERT-Based Classifier Model](https://huggingface.co/bert-base-uncased))

Joint Extractor ([T5-Based Extractor Model](https://drive.google.com/file/d/15OFkWw8kJA1k2g_zehZ0pxcjTABY2iF1/view))

Put all models to `base_models/` for training.

### Model Training

Training model as follows:

1.Run `classifier/Train_generator.py` to train Generator

2.Run `Train_extractor.py` to train Extractor

### AERJE Using

1.Get trained model and run `AERJE_Use.py`, you can quickly use AERJE to extraction API entity and relation from the text.
