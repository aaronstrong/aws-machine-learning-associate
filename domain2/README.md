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
| Unsupervised Learning | Feature Engineering: dimensionality reduction | Drop those columns from a dataset that have a weak relation with the label/target variable: the color of a car when predicting its mileage | Tabular | Principal Component Analysis (PCA) Algorithm |
| Unsupervised Learning | Anomaly Detection | Detect abnormal behavior in applications: spot when an IoT sensor is sending abnormal readings | Tabular | Random Cut Forest (RCF) Algorithm |
| Unsupervised Learning | Clustering or Grouping | Group similar objects/data together: find high-, medium-, and low-spending customers from their transaction histories | Tabular | K-Means Algorithm |
| Unsupervised Learning | Topic Monitoring | Organize a set of documents into topics (not known in advance): tag a document as belonging to a medical category based on the terms used in the document | Text | Latent Dirichlet Allocation (LDA) Algorithm<br>Neural Topic Model (NTM) Algorithm |
| Text Analysis | Speech-to-text | Convert audio files to text; transcribe call center conversations for further analysis | Text | [Sequence-to-Sequence Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/seq-2-seq.html) |
| Text Classification | Assign pre-defined categories to documents in a corpus: categorized books in a library into academic disciplines | Text | Blazing Text Algorithm<br>Text Classification - TensorFlow |
| Image Processing | Image and multi-label classification | Label/tag an image based on the content of the image: alerts about adult content in an image | Image | Image Classification - MXNet |
| Image Processing | Object Detection | Detect people and objects in an image: police review a large photo gallery for a missing person; autonomous vehicle object detection; detecting anomalies in medical images | Image | Image Detection - MXNet<br>Object Detection - TensorFlow |

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

**Latent Dirichlet Allocation (LDA) Algorithm**

* Latent - existing, but not yet developed or manifest
* Allocation - setting apart or earmarking for a particular purpose
* LDA is ideal when you want to categorize text documents without knowing the categories in advance

**[TensorFlow Machine Learning Framework](https://docs.aws.amazon.com/sagemaker/latest/dg/text-classification-tensorflow.html)**

* TensorFlow is an open-source machine learning framework developed by Google.
* Used for neural networks, computer vision, natural language processing (NLP) and more
* SageMaker supplies many pre-trained TensorFlow models which can be modified with fine-tuning

## How to consider interpretability during model selection or algorithm selection

The trade-off between model complexity and interpretability centers on balancing a model’s ability to capture intricate patterns in data versus how easily humans can understand its decision-making.

| Algorithm Class | Intrinsic Interpretability | Pros | Cons |
| --- | --- | --- | --- |
| Linear Models | High | Simple, fast training, direct coefficient, interpretation | May oversimplify complex relationships, performance can be limited |
| Decision Trees | High | Intuitive rule-based logic (If/then statements). Easy to visualize for small trees. | Prone to overfitting, often less accurate than ensembles |
| Ensembles Method | Medium (Feature Important) | High performance, robust | The ensemble of trees makes the overall decision path hard to follow. Provide only global feature importance |
| Neural Networks (Deep Learning) | Low | High performance on complex data (images, text, speech) | "Black box" nature; decisions are highly abstract and non-linear |

### [Fairness, model explainability and bias detection with SageMaker Clarify](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-configure-processing-jobs.html)

You can use Amazon SageMaker Clarify to understand fairness and model explainability and to explain and detect bias in your models. You can configure an SageMaker Clarify processing job to compute bias metrics and feature attributions and generate reports for model explainability. SageMaker Clarify processing jobs are implemented using a specialized SageMaker Clarify container image. 

#### What is fairness and model explainability for machine learning predictions?

Machine learning (ML) models are helping make decisions in domains including financial services, healthcare, education, and human resources. Policymakers, regulators, and advocates have raised awareness about the ethical and policy challenges posed by ML and data-driven systems. Amazon SageMaker Clarify can help you understand why your ML model made a specific prediction and whether this bias impacts this prediction during training or inference. SageMaker Clarify also provides tools that can help you build less biased and more understandable machine learning models.

#### How SageMaker Clarify Processing Jobs Work

You can use SageMaker Clarify to analyze your datasets and models for explainability and bias. You can run a SageMaker Clarify processing job at multiple stages in the lifecycle of the machine learning workflow. SageMaker Clarify can help you compute the following analysis types:

* Partial Dependence Plot (PDP) which can help you understand how much your predicted target variable would change if you varied the value of one feature.
* Pre-, Post-training bias metrics. These metrics can help you understand the bias in your data so that you can address it and train your model on a more fair dataset.


### SageMaker JumpStart

![](https://d2908q01vomqb2.cloudfront.net/da4b9237bacccdf19c0760cab7aec4a8359010b0/2020/11/27/all_solutions-1024x522.png)

* Quickly deploy pre-trained open-source models
* Leverages SageMaker automatic model tuning for hyperparameter tuning
* Allows for automated fine-tuning and deployment
* Generates a Jupyter notebook that is fully customizable
* Allows for fine-tuning using custom data set

#### Pre-trained Models: Deep learning and CNNs

* Object Detection
* Image Classification
* Semantic Segmentation
* Text Classification

These pre-trained models leverage open-source frameworks: TensorFlow, PyTorch and MXNet

#### Pre-trained Models: Classification and Regression

* Tabular classification
* Tabular regrestion

Leverage pre-trained models using LightGBM, CatBoost, XGBoost, and Linear Learner.
All pre-trained models from SageMaker JumpStart support **fine-tuning with custom datasets**

### Amazon Bedrock

* **Key Features**:
  * **Model choice**: Bedrock provides access to a variety of foundation models.
  * **Serverless**: No infrastructure to manage
  * **Customize**: You can customize the models with your data using techniques like fine-tuning and RAG
* **Capabilities**:
  * **Text, Image, and Chat Playground**: Bedrock provides a playground for text, chat, and image models. In these playgrounds you can experiment with models. **Note**: this playground is called *Bedrock Playground*
  * **API**: A detailed API is available that includes actions and their parameters

### Assessing Model Costs

5 Main Drivers of Cost in SageMaker

1. Instance Types
   1. More complex models require higher CPU
2. Data Storage
   1. Depends on the size and outputs of data
   2. More complex models require more size
3. Training Time
   1. Time it takes to train.
   2. Complex models can take months
4. Endpoints
   1. Real-time endpoint or batch
5. Data Transfer
   1. Transfering data from S3 to SageMaker. More data, more cots

## Train and refine models

