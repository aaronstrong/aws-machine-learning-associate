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
| Batch Inference | Offline processing | 100 MB / request | Days | $ |


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

## MLOps

![](https://docs.aws.amazon.com/images/whitepapers/latest/ml-best-practices-public-sector-organizations/images/aws-mlops-framework.png)

### AWS CodePipeline

* CodePipeline is a fully managed, CI/CD service that automates building, testing, and deploying applications and models.
* Integrates with IAM to control access
* supports many AWS services (CodeCommit, S3, ECR) and third party tools like GitHub.
* Each AWS account has a limit of 1000 for the number of pipelines that can be created in a single region
* Max size of input artifacts when stored in GitHub is 1GB

### AWS CodeBuild

* A fully managed, serverless continuous integration service that compiles source code, runs tests,a nd produces deployable software packages.
  * Can build a Docker Image from the `Dockerfile` and `buildspec.yaml`
* No infrastructure to manage, nothing to patch or manage
* Charged based on the number of minutes to take to run
* Works well with CodePipeline, Github, BitBucket and S3
* Uses IAM roles to control access

### AWS CodeDeploy

* A deployment service that automates application deployments to EC2 instances, on-prem instances, servleress Lambda functions, or Amazon ECS services.
* Can deploy application content that runs on a server and is stored in Amazon S3 buckets, GitHub repos, or BitBucket repos. 
* Scales with your infrastructure so you can easily deploy to one instance or thousands.

### AWS CloudFormation

* an Infrastructure as Code service that allows you to model, provision, and manage AWS and third-party resources using JSON or YAML templates.
* Eliminates manual, error-prone configuration
* Infrastructure code is checked into version control systems.

### Amazon EventBridge

* Uses event from CloudWatch, can also schedule tasks
  * Events are state changes
* Continuously track and monitor status changes in SageMaker
* SageMaker model, training job or endpoint state changes
* Data Ingestion events
* Define automated actions that should be invoked when certain events occur
* Initiate Step Functions state machines (workflow) to trigger model trainingw hen new data is ingested.

### Sagemaker Pipelines

![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/pipeline-full.png)

* Purpose built workflow orchestration for machine learning pipelines, integrated with all SageMaker features
* Automate everyone: Data Processing :arrow_right: model training :arrow_right: fine tuning evaluation :arrow_right: deployment :arrow_right: monitoring jobs
* Uses a drag and drop UI to create a series of steps in a directed acyclic graph (DAG). The above is an example of a pipeline DAG:
  * **Data Process**: `AbaloneProcess` runs a preprocessing script on the data used for training.
  * **Training step**: `AbaloneTrain` configures hyperparameters and trains a model from the preprocessed input data.
  * **Processing Step**: `AbaloneEval` evaluates the mode for accuracy.
  * **Condition Step**: `AbaloneMSECond` checks to make sure the mean-square-error results of model evaluation is below a certain limit.
  * **Registered Model**: `AbaloneRegisterModel` step to register the model as a versioned model package group into Amazon SageMaker Model Registry
  * **Create Model**: `AbaloneCreateModel` step to create the model in preparation for batch transformation. In `AbaloneTransform`, SageMaker AI calls a Transform step to generate model predictions on a dataset you specify.

## Automation and integration of data ingestion with orchestration services

Data can be ingested into the machine learning pipeline using an orchestration service. There are two primary services
1. Step Functions
2. AWS CodePipelines

### Step Functions

* Serverless orchestration and workflow service
* Automatically trigger and track each step with logging enabled
* The output of one step is often the input to the next


### AWS CodePipeline

* Automate the continuous delivery of machine learning models

## Deployment strategies and rollback actions 

### [Blue/Green Deployment Methodology](https://docs.aws.amazon.com/whitepapers/latest/blue-green-deployments/introduction.html)

![](https://docs.aws.amazon.com/images/whitepapers/latest/blue-green-deployments/images/blue-green-example.png)

* The fundamental idea behind blue/green deployment is to shift traffic between two identical environments that are running different versions of your application.
* The blue environment represents the current application version serving production traffic.
* In parallel, the green environment is staged running a different version of your application.
* After the green environment is ready and tested, production traffic is redirected from blue to green.
*  If any problems are identified, you can roll back by reverting traffic back to the blue environment.
*  

### Canary Deployment

* Is a type of Blue/Green deployment strategy that is more risk-averse.
* Involves a phased approach in which traffic is shifted to a new version of the application in two increments.
  * The first increment is a small percentage of the traffic, which is referred to as the canary group. Used to test the new version.
  * If successful, the traffic is shifted to the new version in the second increment.
  * In SageMaker AI, you can specify initial traffic distribution by using the `InitialVariantWeight` API.

### Shadow

* In this mode, the new model works alongside an older model or business process, and performs inferences without influencing any decisions.
* This mode is useful as a final check or higher fidelity experiment beforre you promote the model to production.
* Useful if you don't need any user inference feedback

### A/B testing

* This mode is used when the ML practitioners develop new features, but are unsure if the new model will improve business outcomes, like revenue and clickthrough rates.
* Consider an e-commerce site with a business goal to sell as many products as possible. A team member might propose a new review ranking algorithm to improve sales. Using A/B testing, they could roll the old and new algorithms out to different but similar user groups, and monitor the results to see whether users who received predictions form the new model are more likely to make purchases.
* A/B testing also helps gauge the business impact of model staleness and drift. Teams can put new models in production with some recurrence, perform A/B testing with each model, and create an age versus performance chart. This would help the team understand the data drift volatility in their production data.

## Creating automated tests in CI/CD pipelines

Types of Automated tests

* **Unit Tests**:  These tests validate individual units of code in isolation, simulating external dependencies using mocks or emulators. They are low-level, close to the source code, and run quickly within the development environment.
* **Integration Tests**: Focusing on business requirements, these tests check that the system's output meets the specified criteria without necessarily verifying the internal system state.
* **Regression Tests**: Rerun tests to make sure existing features and functions still work and function after a new change has been released.

## Best practices for Retraining a Model- Why Model Retraining is Necessary
### Key Reasons
- **Model Drift**: Real-world data evolves over time, causing deployed models to lose accuracy
- **Concept Drift**: Patterns in data change, degrading model performance
- **Data Decay**: As data ages, model predictions become less reliable
- **Continuous Improvement**: Adaptation to changing data patterns and new business requirements
### Risk of NOT Retraining
Treating model deployment as the final step without ongoing evaluation creates **HIGH RISK** for:
- Significant performance degradation before detection
- Loss of competitive advantage
- Decreased user trust in predictions
- Missed business opportunities
---
## Tools & Services for Model Retraining
### Monitoring & Detection
- **Amazon SageMaker AI Model Monitor** - Continuously monitor model quality, detect data drift, concept drift, bias drift, and feature attribution drift
- **Amazon CloudWatch** - Monitor metrics, create custom dashboards, set alarms for performance thresholds
### Automation & Orchestration
- **AWS Step Functions** - Create automated retraining workflows
- **SageMaker AI Pipelines** - Define and run ML training pipelines
- **Amazon EventBridge** - Trigger retraining pipelines automatically when drift is detected
### Dashboard & Tracking
- **SageMaker AI Model Dashboard** - Centralized interface for tracking deployed models, endpoints, and performance trends
### Human-in-the-Loop
- **Amazon Augmented AI (A2I)** - Route low-confidence predictions to human reviewers for validation and ground truth collection
### Documentation & Knowledge Sharing
- **Amazon QuickSight with GenBI** - Generate visualizations and dashboards automatically
- **Amazon S3** - Store experiment results and operational reports
---
## Implementation Workflow
1. **Monitor** → Establish comprehensive model monitoring with SageMaker Model Monitor
2. **Alert** → Configure CloudWatch alarms to notify teams of performance issues
3. **Track** → Use SageMaker Model Dashboard for visibility across models
4. **Automate** → Create retraining pipelines triggered by drift detection
5. **Validate** → Incorporate human review with A2I for sensitive predictions
6. **Document** → Store learnings and insights for future iterations
7. **Review** → Hold regular stakeholder sessions to discuss improvements
---
## Key Takeaway
Effective model retraining requires **continuous monitoring**, **automated pipelines**, **human validation**, and **documented learnings** to maintain model quality and adapt to changing data patterns over time.
