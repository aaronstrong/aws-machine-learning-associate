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




### The basic architecture of SageMaker Training

The training stage of the full machine learning (ML) lifecycle spans from accessing your training dataset to generating a final model and selecting the best performing model for deployment.

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/sagemaker-training.png)

If using SageMaker AI for the first time and want to find a quick ML solution to train a model on your dataset, considering using a no-code or low-code solution such as
* **SageMaker Canvas** - is a no-code visual interface that enables business analysts and non-technical users to build, train, and deploy machine learing models without writing code.
  * Key Use case: Business predictions (eg: forecasting sales, predicting churn, inventory management)
  * Data Types: Supports tabular, image and text data.
* **SageMaker AutoPiplot** - is an automated machine learning (AutoML) tool that automatically builds, trains,a nd tunes top-performing models based on your tabular data, while offering full transparancy.
  * Key Use Case: Automating the entire machine learning pipeline - preprocessing, training, and tuning - to find the best model.
  * Best For: Users hwo have data science knowledge but want to automate the manual, repetitive steps of ML.
* **SageMaker JumpStart** - Is a machine learning hub that provides pre-trained models, curated soluations, and sample notebooks, allowing users to deploy and fine-tune models with a few clicks.
  * Key Use Case - Rapidly deploing open-source or pre-trained models (like LLMs, Image classification) and fine-tuning them on specific data.
  * Best For: Accessing foundation models and quickly accelerating the development lifecycle.
  * Key Feature: Built-in "model zoo" from sources like TensorFlow, PyTorch, and Hugging Face.

#### Full view of the SageMaker training workflow and features

The following chart shows a high level overview of your actions (in blue boxes) and available SageMaker Training features (in light blue boxes) throughout the training phase of the ML lifecycle.

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/training-main.png)

#### Before Training

There are a number of scenarios of setting up data resources and access you need to consider before training. Refer to the following diagram and details of each before-training stage to get a sense of what decisions you need to make.

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/training-before.png)

* **Prepare data**: Before training, you must have finished data cleaning and feature engineering during the data preparation stage. SageMaker AI has several labeling and feature engineering tools to help you. See Label Data, [Prepare and Analyze Datasets](), [Process Data](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/Processing-1.png), and [Create, Store, and Share Features](https://github.com/aaronstrong/aws-machine-learning-associate/tree/main/aws-services#sagemaker-feature-store) for more information.
* **Choose an Algorithm or framework**: There are different options for algorithms and frameworks.
  * If you prefer low-code implementations of a pre-built algorithm, use one of the built-in algorithms offered by SageMaker
  * If you need more flexibility to customize your model, run your training script using your preferred frameworks and toolkiets within SageMaker AI, use ML Frameworks and Toolkits.
  * To extend pre-built SageMaker AI Docker images as the base image of your own container, see Use pre-built SageMaker AI Docker Images.
  * To bring yoru custom Docker container to SageMaker, see Adapting your own Docker container to work with SageMaker AI. You need to install the sagemaker-training-toolkit to your container.
* **Manage data storage**: Understanding mapping between the data storage ( S3, EFS, FSx) and the training container that runs in the Ec2 compute instance. SageMaker helps map the storage paths and the local paths in the training container. After mapping is done, consider using one of the data transmission modes: File, Pipe and FastFile mode.
* **Set up access to training data**: Use SageMaker Ai domain, a domain user profile, IAM, Amazon VPC, and AWS KMS to meet the requirements of the most security-sensitive orgs.
* **Stream your input data**: SageMaker provides three data input modes: *File*, *Pipe*, *FastFile*. The default input mode is File mode, which loads the entire dataset during initializing the training job.
* **Analyze your data for bias**: Before training, you can analyze your dataset and model for bias against a disfavored group so that you can check that your model learns an unbiased dataset using SageMaker Clarify.
* **Choose which SageMaker SDK to use**: There are two ways to launch a training job in SageMaker AI: using the high-level SageMaker AI PYthon SDK, or using the low-level SageMaker APIs for the SDK for Python (Boto3) or the AWS CLI.


#### During Training

During training, you need to continuously improve training stability, training speed, training efficiency while scaling compute resources, cost optimization, and, most importantly, model performance. 

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/training-during.png)

* **Set up infrastructure**
* **Run a training job from a local code**
* **Tracking Training jobs**
* **Distributed training**
* **Model hyperparameter tuning**
* **Checkpointing and cost savings with Spot instances**

#### After Training

After training, you obtain a final model artifact to use for model deployment and inference. There are additional actions involved in the after-training phase as shown in the following diagram.

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/training-after.png)

* **Obtain baseline model**
* **Examine model performance and check for bias**
* You can also use the Incremental Training funcationality of SageMaker to load and update your model (or fine-tune) with an expanded dataset.
* You can register model training as a step in your SageMaker Pipeline or as part of the Workflow features offered by SageMaker in order to orchestrate the full ML lifecycle.


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

* Object Detection - detects bounding boxes and object labels in an image. It is supervised learning algorithm that supports transfer learning with available pretrained TensorFlow models.
* Image Classification MXNEet - Uses example data with answers (referred to as a *supervised algorithm*). Use this algorithm to classify images.
* Image Classification TensorFlow - uses pretrained TensorFlow Hub models to fine-tune for specific tasks (referred to as a *supvervised algorithm*). use this algorithm to classify images.
* Semantic Segmentation - provides a fine-grained, pixel-level approach to developing computer vision applications.
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


### Underfitting vs Overfitting

![](https://docs.aws.amazon.com/images/machine-learning/latest/dg/images/mlconcepts_image5.png)

Understanding model fit is important for understanding the root cause for poor model accuracy. 

* **Underfitting**
  * Your model is underfitting the <u>training data</u> when the model performs poorly on the training data. This is because the model is unable to capture the relationship between the input examples (often called X) and the target values (often called Y).
  * Might be because the training data is too simple (the input features are not expressive enough) to describe the target well.
  * To improve, by increasing model flexibility by:
    * Add new domain-specific features and more feature Cartesian productions, and change the types of features processing used
    * Decrease the amount of regularization used, which is flatten data
    * add layers to neural network
    * consider a more complex model
* **Overfitting**
  * Overfitting is when the model performs well on the training data but does not perform well on the evaluation data. This is because the <u>model is memorizing the data</u> it has seen and is unable to generalize to unseen examples.
  * Do the following to improve:
    * Feature selection: consider using fewer feature combinations, decrease n-grams size, and decrease the number of numeric attributes bins.
    * **Increase the amount of regularization used**
    * Ensure that your training data set is diverse and representative
    * Apply dropout to your neural network

#### Remedies for Underfitting and Overfitting

| Underfitting | Overfitting |
| --- | --- |
| * **Increase training data quanityt**<br>- Number of data points<br>- Relevent features<br>- Synthentic data<br>**Decrease regularization**<br>- Make weights more 'expressive'<br>**Add max depth, neurons, or layers**<br>**Increase hyperparmeters such as ecphos, batch size, or learning rate may have an effect**<br>**Consider a more complex model**| **Increase training data quality**<br>- Handle outliers in data preprocessing<br>**Increase regularization**<br>- Apply dropout to neural networks<br>- Apply L2 regularization to make model weights less 'expressive'<br>**Reduce max depth, neurons, or layers**<br>**Decrease hyperparameters such as epochs, batch size, or learning state may have an effect**<br> |

### Detecting and Avoiding Catastrophic Forgetting

*This can happen when you're using an Animal Image Recognition model to make predictions on animal images. If your model does great at seeing land animals but not sea creatures, you'll want to supply fine-tuned data of labeled images of sea creatures. In theory the model will see these new creates, but instead catastrophic forgetting happens. Now all land animals are seen as sea creatures.*

* Happens most often when fine-tuning neural networks using traingin data with a novel distribution. Meaning a new data set is introduced that is unlike the original data set to train the model.
* Neural networks will adjust weights to accomodate new training data, potentially impacting performance on previously learned data

#### Technique for mitigating catastrophic forgetting

* Have a baseline. Have a model versioning to compare performance old and new models.
* Rehersal - act of periodically coming back to the original data set for training.
* continual learning frameworks with the use the SageMaker Pipelines

### [Combining multiple training models to improve performance (for example, ensembling, stacking, boosting)](https://aws.amazon.com/blogs/machine-learning/efficiently-train-tune-and-deploy-custom-ensembles-using-amazon-sagemaker/)

![](https://media.geeksforgeeks.org/wp-content/uploads/20251216112827718214/ensemble_learning_.webp)

* In ensembling mode, Autopilot uses stacking to test several combinations of models
* Jumpstart provide <u>pre-trained models</u> while Autopilot helps select algorithms and train models from scratch

![](https://media.geeksforgeeks.org/wp-content/uploads/20250516170016802150/Boosting.webp)
* **Boosting** – Training sequentially multiple weak learners, where each incorrect prediction from previous learners in the sequence is given a higher weight and input to the next learner, thereby creating a stronger learner. Examples include AdaBoost, Gradient Boosting, and **XGBoost**. 

![](https://media.geeksforgeeks.org/wp-content/uploads/20250516170016504785/Bagging.webp)
* **Bagging** – Uses multiple models to reduce the variance of a single model. Examples include Random Forest and Extra Trees. **Random Forest** is a common example of bagging algorithm

![](https://media.geeksforgeeks.org/wp-content/uploads/20250516170017386768/Stacking.webp)
* **Stacking (blending)** – Often uses heterogenous models, where predictions of each individual estimator are stacked together and used as input to a final estimator that handles the prediction. This final estimator’s training process often uses cross-validation.


### Factors for Model performance and size
   * Algorithmic complexity
   * Number of parameters
   * Input data size
   * Quantization
   * Regularization
   * Hyperparameter Tuning

#### Quantization

* Compress models and reduction computation by reducing the precision of numerical values
* Round 32-bit floating point values into lower-preceision data types like 16-bit floating point or 8-bit integers
* This can reduce the memory needs as well as improve the computational efficiency of your model

#### Regularization

* Regularization indirectly effects the size of your model by effectively reducing the weight of parameters
* Two types: L1 and L2
* This technique can be useful when dealing with high-dimensional data sets

#### Hyperparameter Tuning

* Hyperparameters are the tunable settings that determine how a machine learning model learns
* Some settings can affect model size and performance
  - Learning rate
  - Batch Size
  - Number of layers and neurons (neural network)
  - Dropout rate


### Methods to reduce model training time

**Early Stopping**

* Detects validation loss and can stop training when the model begins to overfit


**Distributed Training**

* Distributed training is ideal for deep learning tasks such as computer vision and natural language processing
* Uses SageMaker AI's distributed training libraries or packages such as PyTorch distributedDataParrallel (DDP)
* Perfect use case for FSx for Lustre


**SageMaker AI Managed Warm Pools**

* keep training instances warm for a specified period of time after completing training
* reduct job startup times by 8x

### Benefits of regularization techniques (for example, dropout, weight decay, L1 and L2)

* Regularization mitigates the effect of weighted outliers. In the case of training ML models, this means redistributing the weight your model gives to particular features or neurons
  * **Underfitting**
    * If your model is underfitting, you may need to decrease regularization
  * **Overfitting**
    * If your model is overfitting, you may need to increase regularization

#### L1 Regularization

* L1 identifies the least influential features and sets their model weights to zero
* Useful when you need to reduce the number of features in a large data set
* Not recommended when you want to retain influence from all features in your data

#### L2 Reg

* Proportionally reduces the weights of the largest model weights
* Unlike L1, L2 does not reduce any weights to zero
* ideal if you need to reduce overfitting, but all features are contributing to your predictions

### Hyperparameter tuning techniques (for example, random search, Bayesian optimization)

* Hyperparameters are the tunable settings that determine how a machine learning model learns
* Some settings can affect model seize nad performance:
  - Learning Rate
  - Batch Size
  - Number of layers and neurons (neural networks)
  - Dropout rate

#### Hyperparameter Tuning

* Hyperparameter tuning is an iterative process that involves experimenting with different hyperparameters to optimize for particular outcome, such as maximizing model accuracy, or optimizing the loss function
* This process can be manual or automated
* Common Hyperparameters
  * Epochs
  * Neural network nodes and layers
  * batch size
  * regularization
  * maximum depth
  * learning rate


## Cost and performance

* SageMaker Cost Savings
  * Utilize spot instances
    * Ren unused EC2 capacity from aWS at a discount
    * Great for workloads that are only periodically and that are tolerant to interruptions
    * Can be leveraged in SageMaker by simply enabling Managed Spot Training on your training job 
  * Refine pretrained models with transfer learning
    * Transfer learning is a for of fine-tuning for pre-trained models
    * Take generalized, pretrained mdodels and fine-tune them with domain specific data
    * Retain hyperparameter settings as a starting point when tuning
  * Utilize SageMaker savings plans
    * Pay-in-advance in 1or 3 year terms for SageMaker
    * great for cases where you will have a consistent and predictable timetable

* SageMaker Script Mode
  * Best way to bring your own t raining or inference scripts to use in the SageMaker ecosystem
  * Minimizes code changes needed to run custom workloads on SageMaker's prebuilt containers for frameworks like Scikit-learn, PyTorch, TensorFlow, and XGBoost
  * deploy script mode models for real-time or batch inference
  * Three Layers of Script Mode
    1. Define your own training job, model, and inference process
    2. Modularize your custom workloads and requirements document
    3. Import custom libraries and dependencies

* SageMaker Automatic Model Tuning (AMT)
  * Amazon SageMaker AI automatic model tuning (AMT) finds the best version of a model by running many training jobs on your dataset. AMT is also known as hyperparameter tuning. To do this AMT uses the algorithm and ranges of hyperparameters that you specify. It then chooses the hyperparameter values that creates a model that performs the best, as measured by a metric that you choose.
  * Manages running many training jobs to automate hyperparameter tuning
  * Define your tuning technique and track your objective metric across runs
  * Warm start tuning jobs allow you to initialize using previous jobs as a starting point
    * The results of previous tuning jobs are used to informw hich combinations of hyperparameters to search over in the new tuning job.
    * Identical data and alorightm
    * Transfer learning
    * Types of Warm Start Tuning Jobs
      * *IDENTICAL_DATA_AND_ALGORITH* - The new hyperparmeter tuning job uses the same input data and training images as teh parent tuning jobs. You can change the hyperparameter ranges to search and the maximum number of training jobs that the hyperparmeters tung job launches.
      * *TRANSFER_LEARNING* - The new hyperparemter tuning job can include input data, hyperparameter ranges, maximum number of concurrent training jobs, and maximum number of training jobs that are different than those of its parent hyperparameter tuning jobs.
     
#### 🔑 Quick Comparison Table

| Feature | IDENTICAL_DATA_AND_ALGORITHM | TRANSFER_LEARNING |
|---|---|---|
| Same input data required? | ✅ Yes | ❌ No (can use new/different data) |
| Same training image/algorithm required? | ✅ Yes (minor changes OK) | ❌ No (can use a different version) |
| Can change hyperparameter ranges? | ✅ Yes | ✅ Yes |
| Can change tunable ↔ static hyperparams? | ✅ Yes | ✅ Yes |
| Total static + tunable count must stay same? | ✅ Yes | ✅ Yes |
| Extra response field? | ✅ Yes — `OverallBestTrainingJob` | ❌ No |
| Use case | Expanding search / more training jobs | New data, updated algorithm, or experiments |

#### 📌 IDENTICAL_DATA_AND_ALGORITHM — Cliff Notes

- **Same data, same algorithm** — the new job uses the exact same input data and training image as the parent job(s).
- **Why use it?** You want to:
  - Increase the total number of training jobs (expand the search space)
  - Change hyperparameter ranges or values
  - Flip hyperparameters from tunable → static or static → tunable
- **Algorithm changes:** Minor changes are OK (e.g., logging improvements, different data format support), but you **cannot swap in a completely new version** of the algorithm.
- **Static + tunable count rule:** The total number of static + tunable hyperparameters must stay the same across all parent jobs.
- **Bonus field:** The `DescribeHyperParameterTuningJob` response includes `OverallBestTrainingJob` — a `TrainingJobSummary` of the single best job across this tuning job AND all its parent jobs.

---

#### 📌 TRANSFER_LEARNING — Cliff Notes

- **More flexible** — the new job can have different input data, different hyperparameter ranges, different max concurrent jobs, and different max training jobs vs. the parent job(s).
- **Algorithm flexibility:** The training image/algorithm **can be a different version** from the parent job. (But beware: big changes to the dataset or algorithm can reduce the usefulness of warm start.)
- **Why use it?** You want to:
  - Tune with new or updated data
  - Use a newer version of the training algorithm
  - Experiment with broader changes while still benefiting from prior results
- **Static + tunable count rule:** Same rule applies — total static + tunable hyperparameter count must remain the same across parent jobs.
- **Risk:** If dataset or algorithm changes **significantly affect the objective metric**, warm start tuning may be less effective or even counterproductive.

#### 🧠 Memory Trick

| Type | Think of it as... |
|---|---|
| `IDENTICAL_DATA_AND_ALGORITHM` | "Same recipe, just cook more batches" |
| `TRANSFER_LEARNING` | "New ingredients or a new chef, but learned from last time" |

## Task 2.3: Analyze model performance

<!-- ![](https://www.simplilearn.com/ice9/free_resources_article_thumb/confusion-matrix.JPG) -->

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*XBhArAEOw17ipDex7s5Nqg.jpeg)

### Evaluating Classification Models:
* ![](https://almablog-media.s3.ap-south-1.amazonaws.com/image_14_4f4fc2cf7d.png)
* ![](https://felixaugenstein.com/blog/wp-content/uploads/2023/03/ml-evaluation-classification-1024x474.png)
  * **Accuracy**: is the True Predicitions divided by Total Predictions
    * The accuracy metric minimizes total false predictions without bias toward false positives or false negatives
  * **Precision**:
    * Precision measures the proportion of correct positive predictions over all positive predictions. 
    * The precision metric minimizes false positives without regard for false negatives. Choose this if false positives are very expensive or risky.
    * For example, you may want ot use a high precision model for detecting spam emails
  * **Recall**
    * The recall metric mimizes false negatives without regard for false positives. Choose this if false negatives are very expensive or risky
    * Recall measures the proportion of correct positive predictions over all actual positive values. All actual positive values include boht reviews that correctly predicted as positive (true positive) and reviews that are incorrectly predicted as negative but are actually positive (false negative). To maximize recall means to minimize the false negatives.
    * For example, if your model is screening high risk patients for a deadly condition, you may be willing ot have some false positives to ensure catching as many of the true positives as possible
  * Specificity
    * F1
      * F1 score balances precision and recall. Distinct from accuracy because it still gives valuable results when actual positives are a significant minority

### [Evaluating Binary Classification Models: Receiver Operating Characteristic (ROC) curve](https://docs.aws.amazon.com/machine-learning/latest/dg/binary-model-insights.html)

![](https://docs.aws.amazon.com/images/machine-learning/latest/dg/images/image48b.png)


### Evaluating Regression Model Performance

* **R-Squared**
  * A value from 0 to 1 that describes how well your model explains the variance in the data
  * Generally you want an R-Squared value greater than 0.8, but an R-Squared close to 1 indicates overfitting
* **Root Mean Squared Error (RMSE)**
  * A metric that describes the average magnitude of errors made by the regression model
  * RMSE measures the average distance between predicted and actual values
  * An RMSE close to 0 indicates a very accurate model, but too close 0 could indicate overfitting

## Methods to create performance baselines

Use SageMaker Model Monitor to monitor these 4 areas:

1. Monitor data quality :mag_right:
2. Monitor model performance :dart:
3. Monitor Model bias drift :balance_scale:
4. Monitor feature attribution drift :wrench:
   * Attribute drift can involve a single feature or small group of features dominating predictions.
   * Requires a special baseline call **SHAP baseline** (SHaply Additive exPlanations), which mathematically determines feature attribution.
   * Product violations when feature attribution diverges significantly from your SHAP baseline

* Create a baseline to understand current characteristics
* Schedule monitoring jobs on real-time or batch endpoints
* Deliver metrics to CloudWatch where you can set thresholds

## Comparing the performance of a shadow variant to the performance of a production variant

### [Test models by specifying traffic distribution](https://docs.aws.amazon.com/sagemaker/latest/dg/model-ab-testing.html#model-testing-traffic-distribution)

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/model-traffic-distribution.png)

* Test multiple models by distributing traffic between them, specify the percentage of the traffic that gets routed to each model.

### [Test models by invoking specific variants](https://docs.aws.amazon.com/sagemaker/latest/dg/model-ab-testing.html#model-testing-target-variant)

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/model-target-variant.png)

* Specify the specific version of the model that you want to invoke by providing the value for the `TargetVariant` parameter when you call `invokeEndpoint`

### [Test models using A/B testing]

* Use traffic distribution and invoke specific variants to compare performance metrics between multiple models or model versions

### [Shadow Testing](https://aws.amazon.com/blogs/aws/new-for-amazon-sagemaker-perform-shadow-tests-to-compare-inference-performance-between-ml-model-variants/)

![](https://d2908q01vomqb2.cloudfront.net/da4b9237bacccdf19c0760cab7aec4a8359010b0/2022/11/16/sm-shadow-testing-hiw.png)

* In Shadow Mode, requests are sent to the production variant. The prod variant responds to the user, BUT
* The variant routes a copy of th inference request to a shadow variant, which then responces are stored in S3 bucket
