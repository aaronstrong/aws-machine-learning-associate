# Domain 3: Deployment and Orchestration of ML Workflows

* Task 3.1: Select deployment infrastructure based on existing architecture and requirements
* Task 3.2: Create and script infrastructure based on existing architecture and requirements
* Task 3.3: Use automated orchestration tools to set up continuous integration and continuous delivery (CI/CD) pipelines

## Task 3.1: Select deployment infrastructure based on existing architecture and requirements

Knowledge of:
* Deployment best practices (for example, versioning, rollback strategies)
* AWS deployment services (for example, Amazon SageMaker AI)
* Methods to serve ML models in real time and in batches
* How to provision compute resources in production environments and test environments (for example, CPU, GPU)
* Model and endpoint requirements for deployment endpoints (for example, serverless endpoints, real-time endpoints, asynchronous endpoints, batch inference)
* How to choose appropriate containers (for example, provided or customized)
* Methods to optimize models on edge devices (for example, SageMaker Neo)

Skills in:
* Evaluating performance, cost, and latency tradeoffs
* Choosing the appropriate compute environment for training and inference based on requirements (for example, GPU or CPU specifications, processor family, networking bandwidth)
* Selecting the correct deployment orchestrator (for example, Apache Airflow, SageMaker Pipelines)
* Selecting multi-model or multi-container deployments
* Selecting the correct deployment target (for example, SageMaker AI endpoints, Kubernetes, Amazon Elastic Container Service [Amazon ECS], Amazon Elastic Kubernetes Service [Amazon EKS], AWS Lambda)
* Choosing model deployment strategies (for example, real time, batch)

## Task 3.2: Create and script infrastructure based on existing architecture and requirements

Knowledge of:
* Difference between on-demand and provisioned resources
* How to compare scaling policies
* Tradeoffs and use cases of infrastructure as code (IaC) options (for example, AWS CloudFormation, AWS Cloud Development Kit [AWS CDK])
* Containerization concepts and AWS container services
* How to use SageMaker AI endpoint auto scaling policies to meet scalability requirements (for example, based on demand, time)

Skills in:
* Applying best practices to enable maintainable, scalable, and cost-effective ML solutions (for example, automatic scaling on SageMaker AI endpoints, dynamically adding Spot Instances, by using Amazon EC2 instances, by using Lambda behind the endpoints)
* Automating the provisioning of compute resources, including communication between stacks (for example, by using CloudFormation, AWS CDK)
* Building and maintaining containers (for example, Amazon Elastic Container Registry [Amazon ECR], Amazon EKS, Amazon ECS, by using bring your own container [BYOC] with SageMaker AI)
* Configuring SageMaker AI endpoints within the VPC network
* Deploying and hosting models by using the SageMaker AI SDK
* Choosing specific metrics for auto scaling (for example, model latency, CPU utilization, invocations per instance)

## Task 3.3: Use automated orchestration tools to set up continuous integration and continuous delivery (CI/CD) pipelines

Knowledge of:
* Capabilities and quotas for AWS CodePipeline, AWS CodeBuild, and AWS CodeDeploy
* Automation and integration of data ingestion with orchestration services
* Version control systems and basic usage (for example, Git)
* CI/CD principles and how they fit into ML workflows
* Deployment strategies and rollback actions (for example, blue/green, canary, linear)
* How code repositories and pipelines work together

Skills in:
* Configuring and troubleshooting CodeBuild, CodeDeploy, and CodePipeline, including stages
* Applying continuous deployment flow structures to invoke pipelines (for example, Gitflow, GitHub Flow)
* Using AWS services to automate orchestration (for example, to deploy ML models, automate model building)
* Configuring training and inference jobs (for example, by using Amazon EventBridge rules, SageMaker Pipelines, CodePipeline)
* Creating automated tests in CI/CD pipelines (for example, integration tests, unit tests, end-toend tests)
* Building and integrating mechanisms to retrain models

---


## Model and endpoint requirements for deployment endpoints

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*0NwSyESfaRIyh-Ms)

### [When to use Real-time Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)

* Great for low latency workloads
* Small requests (up to 6MB payload)
* Short running requests (60 seconds)
* Get great performance
* jobs are ran on EC2 instance, which you are paying for.
* EC2 instances run in a VPC, which you get full control over Security Groups
* Suitable for individual requests
  * EC2 instances with a container sits waiting for requests
  * Can use scale-groups to help auto-scale
  * Can scale to zero

### [When to use Serverless Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html#serverless-endpoints-how-it-works-cold-starts)

* Create for the infrequent workloads or intermittent traffic
* Good for small requests (maximum request and response payload is 4MB)
* Short running requests (60 seconds)
* Pay-as-you Go
* No infrastructure to pay for
* Suitable for individual requests
* Workloads must be able to handle a cold start
  * Can mitigate cold starts by purchasing `Provisioned Concurrency`. SageMaker AI keeps the endpoint warm and ready to respond in milliseconds. Even if you're not using them.

### [When to use Asynchronous inference](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference.html)

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/async-architecture.png)

* Use if you have larger requests, up to 1GB payload
* Need longer running requests, up to 60 minutes
* Need near-real time response.
* Endpoints are persistent
* Uses EC2 instance put into an auto-scaling group. That ASG can be scaled to zero
* Results saved in S3

### [When to use Batch Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*gejBBjgAmFxqj8rb25hi4g.png)

* This is used when you need **Offline processing**
  * Like Ad-hoc or scheduled jobs
* Bulk inference for an entire dataset
* Great when working with large datasets like CSV or JSON files
* Maximum payload for each request is 100 MB
* Batch inference is suitable for long running jobs, up to 60 mins per second
* The lowest cost option, of all the options
* No persistent endpoint. You are not paying for any idle time
* No infrastructure to manage. SageMaker provisions and terminates the infrastructure as needed.


## How to choose appropriate containers

<u>Built-in Algorithms for **Supervised** Learning</u>

| Use Case | Problem Type | Built-in Algorithm|
| --- | --- | --- |
| Spam or not spam? | Binary classification<br>Multi-class classification| Linear Learner, KNN, XGBoost, Factorization Machines |
| Predict a numeric value, like house prices | Regression | Linear Learner, KNN, XGBoost, Factorization Machines |
| Predict sales of anew product based on historical sales data | Time-series forecasting | SageMaker DeepAR forecasting |

<u>Built-in Algorithms for **Unsupervised** Learning</u>

| Use Case | Problem Type | Built-in Algorithm|
| --- | --- | --- |
| Detect abnormal behavior, like abnormal sensor readings from IoT devices | Anomaly detection | Random Cut Forest |
| Group similar objects, like categorizing different types of customers | Clustering / grouping | K-Means |
| Discover and idenfity the topics in a set of documents, like categorize a set of documents based on their content | Topic Modeling | Latent Dirichlet Allocation (LDA), Neural Topic Modeling (NTM) |

<u>Built-in Algorithms for **Text Analysis**</u>

| Use Case | Problem Type | Built-in Algorithm|
| --- | --- | --- |
| Group documents into pre-defined categories | Text Classification | BlazingText, Text Classification TensorFlow |
| Convert text to a different language | Translation | Seq-2-Seq |
| Summarize a document | text summarization |  Seq-2-Seq |
| Convert an audio file to a text | Speech to text |  Seq-2-Seq |

<u>Built-in Algorithms for **Image Processing**</u>

| Use Case | Problem Type | Built-in Algorithm|
| --- | --- | --- |
| Label images based on content | Image / multi-label classification | MXNet |
| Detect objects and people in an image | Object detection and classification | MXNet, TensorFlow |
| Categorize and tag every pixel in an image, like self driving car that needs to identifiy objects in its path | Computer Vision | Semantic Segmentation |

## Methods to optimize models on edge devices 

*SageMaker Neo*
* Automatically optimize machine learning models
* To run on EC2 instances or edge devices
* Enabling your model to run faster for no reduction in accuracy

## Evaluating performance, cost, and latency tradeoffs

| Deployment Type | Latency | Max Payload | Timeout | Relative Costs |
| --- | :-: | --- | --- | :-: |
| Real-time hosted endpoint | Real-time | 6 MB / request | 60 seconds | $$$$$ |
| Serverless endpoint | Cold Start | 4 MB / request | 60 seconds | $$$ |
| Asynchronous endpoint | Near real-time | 1 GB / request | 60 minutes | $$<br>auto-scale to 0 |
| Batch Inference | Offline processing | 100 MB / request | 60 minutes | $ |


## Choosing the appropriate compute environment for training

CPU or GPU?

| CPU | GPU |
| --- | --- |
| General compute tasks<br>Serial instruction processing<br>Fewer, more powerful cores<br>General purpose applications| Specialized<br>Parallel instruction processing<br>More, less powerful cores<br>High-performance computing |


<u>GPU-Based AWS Instances</u>
* SKU is EC2 P Instance
* NVIDIA Tensor Core GPUs
* High Performance, ML training, graphics and compute
* ML Workloads run faster on a GPU instance than a CPU instance
* Costs more

<u>AWS Tranium Instances</u>
* Purpose-built, optimized by Amazon
* Train ML Mdoels
* EC Trn Instances
* Networking and Storage
* Cost-effective
* Energy-efficient



## Selecting multi-model or multi-container deployments

### [Multi-model endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html)

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/multi-model-endpoints-diagram.png)

* Multi-model endpoints provide a scalable and cost-effective solution to deploying large number of models. The diagram shows how multi-model endpoints work compared to single-model endpoints.
* Multi-model endpoints are ideal for hosting a large number of models that use the same ML framework on a shared serving container. if you have a mix of frequently and infrequently accessed models, a multi-model endpoint can efficiently serve this traffic with fewer resources and higher cost savings.
* Application should be tolerant of occasional cold start-related latency penalties that occur when invoking infrequent used models.
* Can use the following features:
  * AWS PrivateLink and VPCs
  * Auto Scaling
  * Serial inference pipelines
  * A/B testing

### Selecting the correct deployment target 

* SageMaker Endpoint
  * Running EC2 instances with Docker runtime
  * Under the covers, using ECS
  * Comes with pre-build Docker images with XGBoost, MXNet, TensorFlow
* Elastic Container Service (ECS)
* Elastic Kubernetes Service (EKS)
  * A managed service that makes it easy for you to run Kubernetes on AWS without needing to install and operate your own Kubernetes control plane or worker nodes
* Build and manage your own K8 cluster
* Lambda

### Difference between on-demand and provisioned resources

| On-Demand | Provisioned |
| --- | --- |
| Less predictable traffic | Consistent and predictable |
| Flexible | A commitment is involved |
| No minimum fees | Purhcase reserved capacity|
| No upfront commitment | you pay even if you don't fully use it |
| Pay only for what you use | Increased Performance or throughput |

### Comparing Scaling Policies



### SageMaker Endpoint Auto Scaling Policies

## Target Tracking Scaling (RECOMMENDED)
**What it is**: Automatically adjusts instance count to keep a CloudWatch metric at your target value
**Required Parameters**:
- **Metric**: CloudWatch metric to track (e.g., InvocationsPerInstance)
- **Target Value**: Desired metric value (e.g., 70 invocations/instance/min)
**How it works**: AWS auto-creates alarms and adjusts instances to maintain target
---
## Scaling Limits (MUST SET FIRST)
- **Minimum**: ≥ 1 instance (scales down to this if traffic = 0)
- **Maximum**: ≥ Minimum (no hard cap)
**Config options**: Console, AWS CLI (`--min-capacity`, `--max-capacity`), or API
---
## Cooldown Period (Prevents Over-Scaling)
- **Default**: 300 seconds (5 min)
- **Scale-in cooldown**: Blocks instance deletion temporarily
- **Scale-out cooldown**: Limits new instance creation temporarily
**Adjust if**:
- ⬆️ Increase: Instances adding/removing too fast
- ⬇️ Decrease: Instances not adding fast enough for traffic spikes
---
## Schedule-Based Scaling (Optional)
Time-based scaling at specific times (one-time or recurring)
- **Config method**: CLI or API only (not console)
---
## Quick Exam Facts
✅ Target tracking = metric + target value (simplest approach)
✅ Minimum ≤ Maximum (always)
✅ Minimum ≥ 1 (hard requirement)
✅ Zero traffic auto-scales to minimum only
✅ Cooldown = stability mechanism to prevent rapid scaling
✅ Step scaling = for advanced/complex rules (scale from zero)


### Using Spot Instances
### What It Does
- Uses EC2 Spot instances instead of on-demand to run SageMaker training jobs
- Reduces training costs (up to 90% cheaper)
- AWS manages Spot interruptions automatically
## Key Configuration
- `EnableManagedSpotTraining` = **True**
- `MaxWaitTimeInSeconds` – set larger than `MaxRuntimeInSeconds`
## Important Behavior
- Spot instances can be interrupted, causing longer job times
- **Use checkpointing** to save training progress to S3
- When job restarts, SageMaker loads checkpoint data instead of starting over
## Works With
- Automatic model tuning (hyperparameter tuning)
- Any training job where cost savings matter
## Built-in Algorithm Limit
- Max wait time capped at 3600 seconds (60 minutes)

### Checkpoints in Amazon SageMaker AI - Key Usage Points
### What Checkpoints Do
- Save the state of machine learning models during training
- Snapshots of the model configured by the callback functions of ML frameworks
- Can be used to restart a training job from the last saved checkpoint
## What You Can Do With Checkpoints
- Save model snapshots under training due to unexpected interruption to the training job or instance
- Resume training the model in the future from a checkpoint
- Analyze the model at intermediate stages of training
- Use checkpoints with S3 Express One Zone for increased access speeds
- Use checkpoints with SageMaker AI managed spot training to save on training costs
## How Storage Works
- Training containers on Amazon EC2 instances save checkpoint files under a local directory (default: `/opt/ml/checkpoints`)
- SageMaker provides functionality to copy checkpoints from local path to Amazon S3
- Existing checkpoints in S3 are written to the SageMaker container at the start of the job
- New checkpoints written during training are synced to S3 during training
- If a checkpoint is deleted in the SageMaker container, it's also deleted in the S3 folder

### Configure Sagemaker Endpoints within a VPC Network

*Remember that every AWS service is going to leverage the public endpoint by default*

#### AWS PrivateLink

![](https://docs.aws.amazon.com/images/vpc/latest/privatelink/images/use-cases.png)

* privately connect your VPC to services and resources as if they were in your VPC.
* Public internet access is not required, NAT gateway or internet gateways not needed.
* Create endpoints within your VPC to control and secure traffic
* AWS PrivateLink is what power VPC interface endpoints, which can be used to securely access SageMaker

<u>Interface Endpoints</u>

* Deploys an Elastic Network Interface (ENI) into chosen VPC subnets
  * *ENI is essentially a virtual network interface for a VPC
* The ENI is deployed to a private subnet
* Requires management of an attache security group

### Predefined CloudWatch metrics for SageMaker Autoscaling

You can monitor Amazon SageMaker AI using Amazon CloudWatch, which collects raw data and processes it into readable, near real-time metrics.

Here are some common metrics, every 1 minute:

* `ModelLatency` - The time it takes the model container to process the request and return a response. Measured in microseconds.
* `InovationsPerInstance` - Average number of invocations per instance. Per minute for hte model variant. Variant is the deployed version of your model.
* `CPUUtilization` - Is considered a custom metric. Can scale instances based on the average cpu utilization across all instances.
* **High-resolution** metrics - metrics emitted 10 seconds meaning scaling out is triggered much faster (default is 1 minute).
  * `ConcurrentRequestsperModel`
  * `ConcurrentRequestsPerCopy`