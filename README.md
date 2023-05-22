# Topic-Modelling-Using-RACE-Dataset

## Dataset Summary
**ReAding Comprehension dataset from Examinations (RACE)** is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The dataset is collected from English examinations in China, which are designed for middle school and high school students. The dataset can be served as the training and test sets for machine comprehension.

The **ReAding Comprehension dataset from Examinations (RACE)** dataset is a machine reading comprehension dataset consisting of 27,933 passages and 97,867 questions from English exams, targeting Chinese students aged 12-18. RACE consists of two subsets, RACE-M and RACE-H, from middle school and high school exams, respectively. RACE-M has 28,293 questions and RACE-H has 69,574. Each question is associated with 4 candidate answers, one of which is correct. The data generation process of RACE differs from most machine reading comprehension datasets - instead of generating questions and answers by heuristics or crowd-sourcing, questions in RACE are specifically designed for testing human reading skills, and are created by domain experts.

### Data Fields
The data fields are the same among all splits.

- all
  - example_id: a string feature.
  - article: a string feature.
  - answer: a string feature.
  - question: a string feature.
  - options: a list of string features.
- high
  - example_id: a string feature.
  - article: a string feature.
  - answer: a string feature.
  - question: a string feature.
  - options: a list of string features.
- middle
  - example_id: a string feature.
  - article: a string feature.
  - answer: a string feature.
  - question: a string feature.
  - options: a list of string features.

- Example Data
```
This example was too long and was cropped:

{
    "answer": "A",
    "article": "\"Schoolgirls have been wearing such short skirts at Paget High School in Branston that they've been ordered to wear trousers ins...",
    "example_id": "high132.txt",
    "options": ["short skirts give people the impression of sexualisation", "short skirts are too expensive for parents to afford", "the headmaster doesn't like girls wearing short skirts", "the girls wearing short skirts will be at the risk of being laughed at"],
    "question": "The girls at Paget High School are not allowed to wear skirts in that    _  ."
}
```

### Sources
- [Dataset Link](https://www.cs.cmu.edu/~glai1/data/race/)
- [arxiv.org](https://arxiv.org/abs/1704.04683)
- [paperswithcode.com](https://paperswithcode.com/dataset/race#:~:text=The%20ReAding%20Comprehension%20dataset%20from,Chinese%20students%20aged%2012%2D18.)
