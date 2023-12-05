# AERJE

- Code for ``API Entity and Relation Joint Extraction From Text via Dynamic Prompt-tuned Languange Model``
- Please contact ybsun@jxnu.edu.cn for questions and suggestions.

## Update
- [2023-12-04] update the refactored code.

## Quick Start

Data Link：https://drive.google.com/file/d/1X6pQQhIspNHj2y6GlJaNW0bt3VaQjkXe/view?usp=drive_link

### Download Pretrained Models
Dynamic Prompt Generator ([BERT-Based Classifier Model](https://huggingface.co/bert-base-uncased))

Joint Extractor ([T5-Based Extractor Model](https://drive.google.com/file/d/15OFkWw8kJA1k2g_zehZ0pxcjTABY2iF1/view))

Create 'hf_models' and put all models to 'hf_models/' for training.

### Model Training

Training model as follows:

1.Run `classifier/my_classifier.py` to train Generator

2.Run `main.py` to train Extractor
