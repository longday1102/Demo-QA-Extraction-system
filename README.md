# Question & Answering Extraction system
## Introduction
This source code is made based on [Hugging Face's tutorial on QA Extraction](https://huggingface.co/course/chapter7/7?fw=pt) using Transformer architecture language model. The input to the system is a context and a question, the system will extract the answer in that context.                                                          
## Model
The model used is [bhavikardeshna/xlm-roberta-base-vietnamese](https://huggingface.co/bhavikardeshna/xlm-roberta-base-vietnamese), which is a language model based on [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta), trained on Vietnamese dataset.                                                                                    
The model is described in [Cascading Adaptors to Leverage English Data to Improve Performance of
Question Answering for Low-Resource Languages](https://arxiv.org/pdf/2112.09866v1.pdf) paper.
## Datasets
The dataset used is [UIT-ViQuAD](https://github.com/windhashira06/Demo-QA-Extraction-system/tree/main/Dataset). This dataset comprises over 23,000 human-generated question-answer pairs based on 5,109 passages of 174 Vietnamese articles from Wikipedia. However in processing, I eliminated more than 3000 questions with no answers.
## Evaluation
The dataset after processing is divided with test size is 0.06. Below are the evaluation results on test set:                                           
| EM | F1-SCORE |
|:----:|:---------:|
| 52.38 | 77.67 |
## Test
Below are some test results:                                          

![longhoang06_fine-tuned-viquad-hgf-·-Hugging-Face-and-2-more-pages-Personal-Microsoft_-Edge-2023-03-1](https://user-images.githubusercontent.com/121651344/225209165-56f4e858-91ae-4d01-b7e8-51728f85d792.gif)
<p align="center">
<b>Test.1</b>
</p>

![longhoang06_fine-tuned-viquad-hgf-·-Hugging-Face-and-3-more-pages-Personal-Microsoft_-Edge-2023-03-1](https://user-images.githubusercontent.com/121651344/225249920-d70aa8df-f131-424e-8b55-0fe981ce2e2c.gif)
<p align="center">
<b>Test.2</b>
</p>

![longhoang06_fine-tuned-viquad-hgf-·-Hugging-Face-and-3-more-pages-Personal-Microsoft_-Edge-2023-03-1_3](https://user-images.githubusercontent.com/121651344/225250177-dbea9b6a-a668-4a75-a727-cd25247f3c49.gif)
<p align="center">
<b>Test.3</b>
</p>

Relatively good 😅


