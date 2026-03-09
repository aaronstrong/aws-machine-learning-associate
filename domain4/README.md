# Domain 4: ML Solution Monitoring, Maintenance, and Security

Tasks
* Task 4.1: Monitor model inference
* Task 4.2: Monitor and optimize infrastructure and costs
* Task 4.3: Secure AWS resources

## Task 4.1: Monitor model inference
Knowledge of:
* Drift in ML models
* Techniques to monitor data quality and model performance
* Design principles for ML lenses relevant to monitoring

Skills in:
* Monitoring models in production (for example, by using Amazon SageMaker Model Monitor)
* Monitoring workflows to detect anomalies or errors in data processing or model inference
* Detecting changes in the distribution of data that can affect model performance (for example, by using SageMaker Clarify)
* Monitoring model performance in production by using A/B testing

## Task 4.2: Monitor and optimize infrastructure and costs
Knowledge of:
* Key performance metrics for ML infrastructure (for example, utilization, throughput, availability, scalability, fault tolerance)
* Monitoring and observability tools to troubleshoot latency and performance issues (for example, AWS X-Ray, Amazon CloudWatch Lambda Insights, Amazon CloudWatch Logs Insights)
* How to use AWS CloudTrail to log, monitor, and invoke re-training activities
* Differences between instance types and how they affect performance (for example, memory optimized, compute optimized, general purpose, inference optimized)
* Capabilities of cost analysis tools (for example, AWS Cost Explorer, AWS Billing and Cost Management, AWS Trusted Advisor)
* Cost tracking and allocation techniques (for example, resource tagging)

Skills in:
* Configuring and using tools to troubleshoot and analyze resources (for example, CloudWatch Logs, CloudWatch alarms)
* Creating CloudTrail trails
* Setting up dashboards to monitor performance metrics (for example, by using Amazon QuickSight, CloudWatch dashboards)
* Monitoring infrastructure (for example, by using Amazon EventBridge events)
* Rightsizing instance families and sizes (for example, by using SageMaker AI Inference Recommender and AWS Compute Optimizer)
* Monitoring and resolving latency and scaling issues
* Preparing infrastructure for cost monitoring (for example, by applying a tagging strategy)
* Troubleshooting capacity concerns that involve cost and performance (for example, provisioned concurrency, service quotas, auto scaling)
* Optimizing costs and setting cost quotas by using appropriate cost management tools (for example, AWS Cost Explorer, AWS Trusted Advisor, AWS Budgets)
* Optimizing infrastructure costs by selecting purchasing options (for example, Spot Instances, OnDemand Instances, Reserved Instances, SageMaker AI Savings Plans)

## Task 4.3: Secure AWS resources
Knowledge of:
* IAM roles, policies, and groups that control access to AWS services (for example, AWS Identity and Access Management [IAM], bucket policies, SageMaker Role Manager)
* SageMaker AI security and compliance features
* Controls for network access to ML resources
* Security best practices for CI/CD pipelines

Skills in:
* Configuring least privilege access to ML artifacts
* Configuring IAM policies and roles for users and applications that interact with ML systems
* Monitoring, auditing, and logging ML systems to ensure continued security and compliance
* Troubleshooting and debugging security issues
* Building VPCs, subnets, and security groups to securely isolate ML systems

---

## ML Solution Monitoring, Maintenance, and Security

### What is drift?

*  A term used to describe how performance of a machine learning model in production slowly gets worse over time. This can happen for a number of reasons:
   * Data Drift
     * Examples like Season Changes to data
     * Changes in weather conditions
     * New products are added
   * Model Drift
     * 
   * Bias Drift
     * use something like SageMaker Clarify to help reduce bias drift
   * Feature Attribution Drift

* Tools to monitor drift: SageMaker Model Monitor
  * SageMaker Model Monitor enables you to capture the input, output and meatadata for the invocations of the models that you deploy


### Monitoring Data & Model Quality

| Data Quality Monitoring | Model Quality MOnitoring |
| --- | --- |
| Enable data capture | Enable data capture |
| Create a baseline| Create a baseline |
| Define and schedule monitoring jobs | Define and schedule monitoring jobs |
| View data quality metrics | Ingest ground truth labels<br>Bring in humans, human in the loop |
| Integrate monitoring with CloudWatch | Integrate monitoring with CloudWatch | 


## Design principles for ML lenses relevant to monitoring

<u>[Well-Architected Machine Learning Lifecycle](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lifecycle.html</u>

![](https://docs.aws.amazon.com/images/wellarchitected/latest/machine-learning-lens/images/ml-lifecycle.png)

* Business goal identification
* ML problem framing
* Data Processing
* Model development
* Deployment
* Monitoring
  * Model Explainability
  * Detect Drift
  * Model update pipeline



### Monitoring workflows to detect anomalies or errors in data processing or model inference

* Random Cut Forest (RCF) for Anomaly Detection
  * Algorithm developed by Amazon
    * Create a forest of trees where each tree is obtained using a partition of a sample of the training data. Each tree is given such a partition and organizes that subset of point into a *k-d* tree.
  * Unsupervised
    * Clustering of data
  * Assigns anomaly score to each data point
  * Used for real-time anomaly detection
  * RCF inputs:
    * RecordIO-protobuf
    * CSV
  * RCF Training modes:
    * File Mode - Upload a file
    * Pipe Mode - Pipe the data straight into the RCF algorithm on the fly. shorten the startup and download time, reduce startup time
  * Use Cases
    * Cost Anomaly detection - increases in spending
    * Fraudulent activity detection - retail or banking, like someone getting a hold of credit cards or rising costs
    * Manufacturing quality
    * Predictive Maintenance - Sensor data, think of NASA and looking at temperatures sensor data

#### Types of Anomalies

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/RCF1.jpg)
*Example of a time series data - data is clustered together with one outlier*

* Breaks in periodicity
* Unclassified data points
* Unexpected spikes in time series data

### Detecting changes in the distribution of data that can affect model performance

<u>Key ML terms for SageMaker Clarify</u>

* **Fairness** - The process of ensuring that ML models treat all individuals equally, particularly when making decisions that impact groups based on protected characteristics
* **Explainability** - The ability to explain a machine learning model's output and decision-making process in a way that humans can understand
* **Bias** - When an algorithm produces results that are systemically prejudiced due to erroneous assumptions in the machine learning process

<u>Bias Detection with SageMaker Clarify</u>

* Detect bias in and help explain your model predictions
* Identify types of bias in pre-training data
* Identify types of data in post-training data

#### How SageMaker Clarify Processing Jobs Work

You can use SageMaker Clarify to analyze your datasets and models for explainability and bias. A SageMaker Clarify processing job uses the SageMaker Clarify processing container to interact with an Amazon S3 bucket containing your input datasets. You can also use SageMaker Clarify to analyze a customer model that is deployed to a SageMaker AI inference endpoint.

The following graphic shows how a SageMaker Clarify processing job interacts with your input data and optionally, with a customer model. This interaction depends on the specific type of analysis being performed. The SageMaker Clarify processing container obtains the input dataset and configuration for analysis from an S3 bucket. For certain analysis types, including feature analysis, the SageMaker Clarify processing container must send requests to the model container. Then it retrieves the model predictions from the response that the model container sends. After that, the SageMaker Clarify processing container computes and saves analysis results to the S3 bucket.

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/clarify/clarify-processing-job.png)

You can run a SageMaker Clarify processing job at multiple stages in the lifecycle of hte machine learning workflow.

* Pre-training bias metrics. These metrics can help you understand in your data so that you can address it and train your model on a more fair dataset.
* Post-training bias metrics. These metrics can help you understand any bias introduced by an algorithm, hyperparameters choices, or any bias that wasn't apparaent earlier in the flows.
* Shapley values, which can help you understand what impact your feature has on what your model predicts.
* Patrial dependence plots (PDP), which can help you understand how much your predicted target variable would change if you varied the value of one of the feature.

### Monitoring model performance in production by using A/B testing

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/model-traffic-distribution.png)

A/B testing compares two or more versions of a design to determine which performs best.

You then specify an amount of traffic to go to a specific model.


## Monitoring and observability tools to troubleshoot latency and performance issues

### Tools to detect and mitigate Model Drift

* AWS X-Ray
* AWS CloudWatch Lambda insights
* CloudWatch Log Insights

#### X-Ray

* Collects data about requests that your application serves and provides tools that you can use to view, filter, and gain insights into that data to identify issues and opportunities for optimization.
* Collects traces from your app, in addition to AWS services your application uses that are already integrated with X-ray
* Think **end-to-end**, and **traces**. X-ray can collect traces from SageMaker

#### CloudWatch Basics

* Monitor SageMaker using CloudWatch, which collects raw data and porcesses it into readable, near real-time metrics
* Set alarms to watch for certain thresholds and take actions when those thresholds are met.

