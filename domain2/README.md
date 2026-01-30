# Domain 2: ML Model Development

* Task 2.1: Choose a modeling approach
* Task 2.2: Train and refine models
* Task 2.3: Analyze model performance

---

## Task 2.1: Choose a modeling approach

Knowledge of:
* Capabilities and appropriate uses of ML algorithms to solve business problems
* How to use AWS artificial intelligence (AI) services (for example, Amazon Translate, Amazon Transcribe, Amazon Rekognition, Amazon Bedrock) to solve specific business problems
* How to consider interpretability during model selection or algorithm selection
* Amazon SageMaker AI built-in algorithms and when to apply them

Skills in:
* Assessing available data and problem complexity to determine the feasibility of an ML solution
* Comparing and selecting appropriate ML models or algorithms to solve specific problems
* Choosing built-in algorithms, foundation models, and solution templates (for example, in SageMaker JumpStart and Amazon Bedrock)
* Selecting models or algorithms based on costs
* Selecting AI services to solve common business needs

## Task 2.2: Train and refine models

Knowledge of:
* Elements in the training process (for example, epoch, steps, batch size)
* Methods to reduce model training time (for example, early stopping, distributed training)
* Factors that influence model size
* Methods to improve model performance
* Benefits of regularization techniques (for example, dropout, weight decay, L1 and L2)
* Hyperparameter tuning techniques (for example, random search, Bayesian optimization)
* Model hyperparameters and their effects on model performance (for example, number of trees in a tree-based model, number of layers in a neural network)
* Methods to integrate models that were built outside SageMaker AI into SageMaker AI

Skills in:
* Using SageMaker AI built-in algorithms and common ML libraries to develop ML models
* Using SageMaker AI script mode with SageMaker AI supported frameworks to train models (for example, TensorFlow, PyTorch)
* Using custom datasets to fine-tune pre-trained models (for example, Amazon Bedrock, SageMaker JumpStart)
* Performing hyperparameter tuning (for example, by using SageMaker AI automatic model tuning [AMT])
* Integrating automated hyperparameter optimization capabilities
* Preventing model overfitting, underfitting, and catastrophic forgetting (for example, by using regularization techniques, feature selection)
* Combining multiple training models to improve performance (for example, ensembling, stacking, boosting)
* Reducing model size (for example, by altering data types, pruning, updating feature selection, compression)
* Managing model versions for repeatability and audits (for example, by using the SageMaker Model Registry)

## Task 2.3: Analyze model performance

Knowledge of:
* Model evaluation techniques and metrics (for example, confusion matrix, heat maps, F1 score, accuracy, precision, recall, Root Mean Square Error [RMSE], receiver operating characteristic [ROC], Area Under the ROC Curve [AUC])
* Methods to create performance baselines
* Methods to identify model overfitting and underfitting
* Metrics available in SageMaker Clarify to gain insights into ML training data and models
* Convergence issues

Skills in:
* Selecting and interpreting evaluation metrics and detecting model bias
* Assessing tradeoffs between model performance, training time, and cost
* Performing reproducible experiments by using AWS services
* Comparing the performance of a shadow variant to the performance of a production variant
* Using SageMaker Clarify to interpret model outputs
* Using SageMaker Model Debugger to debug model convergence

---

# My Notes

![](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2025/11/17/ml-17163-img.png)

![](https://d2908q01vomqb2.cloudfront.net/fc074d501302eb2b93e2554793fcaf50b3bf7291/2022/02/10/Figure-2.-Managed-AWS-AI-services.jpg)


## Choose a modeling approach
### Vision

| Subject | Overview | Use Cases |
| --- | --- | --- |
| Amazon Rekognition | * Easy API that can analyze image or video files sotred in S3<br>*Add features that detect objects, text, unsafe content, analyze images/videos, compare faces | * User verification<br>* Cataloging<br>* People Counting<br>* Public Safety |
| Amazon Textract | * Add document and text detection and analysis to your apps<br>* Extract text, forms, and tables from documents with structured data<br>* Process invoices and reciepts<br>* Process ID docs like drivers licenses| - |

### Language

| Subject | Overview | Use Cases |
| --- | --- | --- |
| Amazon Comprehend | Analyze text data, uses natural language processing (NLP<b>*Comprehend can find insights from Built-in classifications like spam detection, Sentiment Analysis, Entity Recognition, Idenity and redact PII) | - |
| Amazon Translate | A text translation service to provide high-quality translation on demand | * Enable multilingual user experiences in your apps<br>* Process and manage your company's incoming data like analyze text or social media feeds |

### Speech

| Subject | Overview | Use Cases |
| --- | --- | --- |
| Amazon Polly | Text-to-speech | - |
| Amazon Transcribe | Speech-to-text | - |

### Chatbots

| Subject | Overview | Use Cases |
| --- | --- | --- |
| Amazon Lex | Same technology that powers Alexa. Converts speech to text to build chat bots | * Chatbots<br>* Voice Assistance |

### Forecasting

| Subject | Overview | Use Cases |
| --- | --- | --- |
| Amazon Forecast | - | - |

### Search

| Subject | Overview | Use Cases |
| --- | --- | --- |
| Amazon Kendra | Intelligent enterprise search service. Uses machine learning to search unstructured data. Can work with S3 and Lex | - |


### Problem framing your ML solution

* What is the nature of your problem?
* What kind of data is available for training
* Which machine learning algorithms are most appropriate for your use case?

**Supervised Learning**

This model trains on data that has the input and correct outputs (labels). Relies heavily on correct labels. use cases are image classification and spam detection. 

**Unsupervised Learning (Unlabeled Data)**

The model trains on unlabeled input data to find patterns and groups. Use if your inputs that are not labeled. Use cases are csutomer segmentation and anomaly detection.

**Semi-Supervised Learning**

The model creates its own labels based on the data throught tasks like predicting missing information. use if you have a small amout of partiially labeled data with a large amount of unlabeled data.

**Reinforcement Learning**

Enforcing models to make decisions. The algorithm of this method helps the model learn based on feedback. Use cases are self-driving cars.

### [Amazon SageMaker AI built-in algorithms and when to apply them](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)

| Learning paradigm or domain | Problem Types | Examples & Use cases | Data input format | Built-in Algorithms |
| --- | --- | --- | --- | --- |
| Pre-trained models and pre-built solution templates | Image Classification<br>Tabular Classification<br>Text Classification<br>Image Embedding | Here | - | - |
| Supervised Learning | Time-series Forecasting | Based on historical data for behavior; predict future behavior: predict sales on a new product | Tabular | [SageMaker AI DeepAR forecasting algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html) |
| Supervised Learning | Embeddings: convert high dimensinal objects into dimensional space | Improve the data embeddings of the high-dimensional objects: identify duplicate support tickets or find the correct routing based on similiarity of text in the tickets | Tabular | Object2Vec |
| Supervised Learning | Regression | Predict a numeric/continuous value: estimate the value of a house | Tabular | Factorization machines algorithm<br>**K-Nearest Neighbors (k-NN)**<br>**LightGBM**<br>TabTransformer<br>**XGBoot**<br>**Linear Learner** |
|Supervised Learning | Binary/multi-class classification | Predict if an item belongs to a category: email spam filter | Tabular | Factorization machines algorithm<br>k-nearest neighbors<br>LightGBM<br>TabTransformer<br>XGBoost<br>Linear Learner |

**[K-Nearest Neighbors (k-NN)](https://docs.aws.amazon.com/sagemaker/latest/dg/k-nearest-neighbors.html)**

* Determines a value based on the average of values from its nearest neighbors.
* Ideal for low-dimensional data where the relationship between features and output is complex
* Can be used for regression for classification

**LightGBM**

* Can be used for regression and classification
* Efficently handles high-dimensional data sets with millions of rows

**XGBoost**

* Stands for eXtreme Gradient Boosting
* Gradient boosting involves generating predictions by combining estimates from serveral simpler models
* Efficient and robust algorithm for regression or classification problems
* Highly tunable hyperparameter setting
* is a very popular open source tool

**Linear Learner**

* Linear learner is a simple algorithm for regression and classification problems.
* Applies a linear function to define a trend or thresholds
