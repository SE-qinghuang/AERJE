# AERJE

- Code for [``API Entity and Relation Joint Extraction From Text via Dynamic Prompt-tuned Languange Model``](https://www.semanticscholar.org/reader/fa9d49f32440aff7417ce46419d1073239b58b5b)
- The Data Link: [https://anonymous.4open.science/r/AERJE-F7D5/README.md](https://drive.google.com/file/d/1X6pQQhIspNHj2y6GlJaNW0bt3VaQjkXe/view?usp=sharing)
- Please contact ybsun@jxnu.edu.cn for questions and suggestions.
- The project is under construction...

## Update
....

## Requirements
- Python (verified on 3.8)
- CUDA (verified on 11.1/10.2)
## Quick Start

### Data Collator
Run `dataset_processing/spider.py` to fetch the raw unlabeled data from Stackoverflow.

### Datasets Processing
1.Data Increasing

Run `dataset_processing/data_increasing.py` enhance labeled data

2.Convert data to training Generator

Run `dataset_processing/Convert_Classifier` convert labeled data to data that training Generator

3.Convert data to training Extractor

Run `dataset_processing/Convert_AERJE.py` convert labeled data to data that training Extractor

### Pretrained Models
You download the pre-trained models

AERJE-extractor-en-base ([Bert based model](https://huggingface.co/bert-base-uncased))

AERJE-generator-en-base ([T5-base](https://drive.google.com/file/d/12Dkh6KLDPvXrkQ1I-1xLqODQSYjkwnvs/view?pli=1) / [T5-large](https://drive.google.com/file/d/12Dkh6KLDPvXrkQ1I-1xLqODQSYjkwnvs/view?pli=1))

Put all models to `base_models/` for training.

### Model Training

Training model as follows:

1.Run `classifier/Train_generator.py` to train Generator

2.Run `Train_extractor.py` to train Extractor

### AERJE Using

1.Get trained model and run `AERJE_Use.py`, you can quickly use AERJE to extraction API entity and relation from text.

## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only.
Any commercial use should get formal permission first.
