# Xiao-Shih

This is the code repository for:

H. Hsu and N. Huang, “Xiao-Shih: A Self-enriched Question Answering Bot With Machine Learning on Chinese-Based MOOCs,” <I>IEEE Trans. Learning Technologies</I>. (Under Review)

## Spreading Question Similarity (SQS)
SQS algorithm was proposed to compute question similarities based on keyword networks. As the name suggests, this algorithm spreads the degree of relationship between the most relevant keywords by iterating the neighbors on keyword networks. Because of this, vectors will not only be generated with existing keywords but also existing keywords will find other relevant keywords and integrate their similarities into vectors.

- [SQS.ipynb](https://github.com/PyDataScience/Xiao-Shih/blob/main/SQS.ipynb)

## Xiao-Shih: Question Answering Bot
Xiao-Shih will predict whether the new question and the archived question are duplicates or not by ML. If yes, Xiao-Shih may respond the answer of the archived question to the learner.

- [QA.ipynb](https://github.com/PyDataScience/Xiao-Shih/blob/main/QA.ipynb)
