# In-Scope AWS Services

### Table of Contents

* Analytics
  * [AWS Glue](#aws-glue)
  * [AWS Glue Databrew](#aws-glue-databrew)
  * [Amazon EMR]()
  * [AWS Lake Formation](#aws-lake-formation)
  * [Amazon Kinesis](#amazon-kinesis)
    * [Amazon Kinesis Data Streams](#kineses-data-streams)
    * [Amazon Data Firehose](#amazon-data-firehose)
    * [Amazon Managed Service for Apache Flink](#managed-service-for-apache-flink)
* Machine Learning
  * [Amazon SageMaker](#amazon-sagemaker)
  * [Amazon Mechanical Turk]()
  * [Amazon TextTract]()
* Storage
  * [Amazon S3](#amazon-s3)
  * [Amazon EBS](#amazon-elastic-block-store-amazon-ebs)
  * [Amazon EFS](#amazon-elastic-file-system-amazon-efs)
  * [Amazon FSx](#amazon-fsx)
  * [Amazon Storage Gateway](#aws-storage-gateway)
* Migration Services
  * [AWS DataSync](#aws-datasync)
  * [AWS DMS](#aws-database-migration-services-aws-dms)



## Analytics

![](https://docs.aws.amazon.com/images/decision-guides/latest/analytics-on-aws-how-to-choose/images/analytics-services.png)


### AWS Glue

* AWS Glue is a family of integrated services for serverless data discovery, preparation, and ETL
* Define ETL jobs as Python shell jobs or Apache Spark jobs
* Define low-code ETL jobs using AWS Glue Studio
* Charged per DPU-hour. A standard DPU provides 4 vCPU and 16 GB of Memory
* Run event-driven or scheduled batch jobs
  * **Python Shell jobs**
    * uses 1 DPU or a fraction of a DPU (equivalent to 1 GB memory)
    * Recommended for simple transformations on data sets less than 10GB
  * **Spark Jobs**
    * Allocate 2 or 100 DPUs, or for Glue 2.0 jobs specify types and number of workers.
    * Recommended for more memory intensive jobs on larger data sets

### AWS Glue DataBrew

* A simpler data transformation tool for cleaning and normalizing data
* Designed for nontechnical data analysts
* Assemble predefined no-code data transformations in visual interface
* Visually explore data sets
* Map data lineage
* Reuse defined transformations in automated processes

### Amazon Kinesis

A platform for streaming data on AWS that makes it easy to load and analyze streaming data. Amazon Kinesis also enables you to build custom streaming data applications for specialized needs. With Kinesis you can ingest real-time data such as application logs, website clickstreams, IoT telemetry, and more into your databases, data lakes, and data warehouses. Kinesis enables you to process and analyze data as it arrives and respond in real-time instead of having to wait until your data is collected before the processing can begin.

There are 4 pieces to Kinesis:

* [Amazon Kinesis Data Streams](#kineses-data-streams) - enables you to build custom applications that process or analyze streaming data
* Amazon Kinesis Video Streams - enables you to build custom applications that process or analyze streaming video
* [Amazon Data Firehose](#amazon-elastic-block-store-amazon-ebs) - enables you to deliver real-time streaming data to AWS destinations like S3, Redshift, OpenSearch Services, and Splunk. Configure your data producers to send data to Amazon Data Firehose, and it automatically delivers the data to teh destinations that you specified. Can configure Firehose to transform your data before delivering it.
* [Amazon Managed Service for Apache Flink](#managed-service-for-apache-flink) - enables you to process and analyze streaming data with standard SQL or with Java

#### Kineses Data Streams

<img src="https://docs.aws.amazon.com/images/streams/latest/dev/images/architecture.png" alt="Alt text" width="600" height="300">


* Fully managed service to ingest data streams
* Streams are split into shards (a shard is a uniquely identified sequence of data records in a stream. A stream is composed of 1 or more shards)
  * Write up to 1 MB or 1000 records per second
  * Read up to 2 MB or 2000 records per second for each shard
  * If Kinesis is under performing, increase the number of shards
* Default limit of 10,000 shards per stream, but there is technically no upper limit

#### Amazon Data Firehose

For Amazon S3 destinations, streaming data is delivered to your S3 bucket. if data transformation is enabled, you can optionally back up source data to another S3 bucket.

<img src="https://docs.aws.amazon.com/images/firehose/latest/dev/images/fh-flow-s3.png" width="600" height="300">

For Amazon Redshift destinations, streaming data is delivered to your S3 bucket first. Data Firehose then issues an Amazon Redshift **COPY** command to load data from your S3 bucket to your Redshift cluster. If data transformation is enabled, you can opitonaly back up source data to another S3 bucket.

<img src="https://docs.aws.amazon.com/images/firehose/latest/dev/images/fh-flow-rs.png" width="600" height="300">

* Deliver to integrated services like S3, Redshift, or Amazon OpenSearch service
* Deliver to popular applications like Splunk, Snowflake, or a custom HTTP endpoint
* Convert data to Parquet or ORC
* Integrate with Lambda for custom transformations
* Dynamically partition data delivered to S3

#### Managed Service for Apache Flink

![](https://d1.awsstatic.com/Picture1.f8c5ecd75aae8cd14f6041541b55d5c5985487a6.f8c5ecd75aae8cd14f6041541b55d5c5985487a6.jpg)

Apache Flink is an open-source, distributed engine for stateful processing over unbound (streams) and bounded (batches) data sets. Stream processing applications are designed to run continuously, with minimal downtime, and process data as it is ingested. Flink is designed for low latency processing, performing computations in-memory, for high availability, removing single point of failures, and to scale horizontally.

**Why use Apache Flink**

* **Event-driven applications**, ingesting events from one or more event streams and executing computations, state updates or external actions. Stateful processing allow implementing logic beying the Single Message Transformation, where the resutls depend on the history of the ingested events.
* **Data Analytics applications**, extracting information and insights from data. 
* **Data pipeline applications**, transforming and enriching data to be moved from one data storage to another. Traditionally, ETL is executed periodically, in batches. With Apache Flink, the process can operate continuously, moving the data with low latency to their destinations.

* A way to live process and analyse data as it's streamed
* Interactively query real-time data and generate continuous insights
* Detect outliers and threshold breaches as early as possible

### Amazon Managed Streaming for Apache Kafka (MSK)

* Create Apache Kafka clusters from scratch or deploy your existing Kafka cluster to AWS
* Optimized for capturing log and event streams
* Native integrations with Kinesis family, EC2, Lambda, Redshift, and others
* Typically only recommended over Kineses Data Streams in cases where your organization or application is already using Kafka

### [AWS Lake Formation](https://docs.aws.amazon.com/lake-formation/latest/dg/what-is-lake-formation.html)

AWS Lake Formation helps you centrally govern, secure, and globally share data for analytics and machine learning. With Lake Formation, you can manage fine-grained access control for your data lake data on Amazon Simple Storage Service (Amazon S3) and its metadata in AWS Glue Data Catalog.

Lake Formation provides its own permissions model that augments the IAM permissions model. Lake Formation permissions model enables fine-grained access to data stored in data lakes as well as external data sources such as Amazon Redshift data warehouses, Amazon DynamoDB databases, and third-party data sources. Lake Formation permissions are enforced using granular controls at the column, row, and cell-levels across AWS analytics and machine learning services, including Amazon Athena, Amazon Quick Suite, Amazon Redshift Spectrum, Amazon EMR, and AWS Glue.

![](https://d2908q01vomqb2.cloudfront.net/b6692ea5df920cad691c20319a6fffd7a4a766b8/2019/07/10/B.jpg)

**How it works**

AWS Lake Formation provides a relational database management system (RDBMS) permissions model to grant or revoke access to Data Catalog resources such as databases, tables, and columns with underlying data in Amazon S3. The easy to manage Lake Formation permissions replace the complex Amazon S3 bucket policies and corresponding IAM policies.

![](https://docs.aws.amazon.com/images/lake-formation/latest/dg/images/lf-workflow.png)

**AWS Service integration with Lake Formation**


| AWS Service | Integration Details |
| --- | --- |
| AWS Glues | AWS Glue and Lake formation share the same Data Catalog |
| Amazon Athena | Use Lake Formation to allow or deny permissions to read data in S3. |
| Amazon Redshift Spectrum | When Amazon Redshift users create an external schema on a database in the AWS Glue Data Catalog, they can query only the tables and columns in that schema in which they have Lake Formation permissions. |
| Amazon Quick Suite Enterprise Edition | When an Amazon Quick Suite Enterprise Edition user queries a dataset in an Amazon S3 location, the user must have the Lake Formation `SELECT` permission on the data. |
| Amazon EMR | You can integrate Lake Formation permissions when you create an Amazon EMR cluster with a runtime role. |

**General, Example, Architecture**

High-level example of an AWS Lake formation architecture: Customers ingest data from multiple sources into their data lake. Happy HealthCare’s marketing team receives data from multiple EHR providers to identify count of patients within a specific geography. Since data is coming from different EHR providers, and SSN number is optional, it is getting difficult for Happy HealthCare to identify the unique customers.

Happy HealthCare decided to use AWS Lake Formation ML Transforms to identify the potential duplicates in data. They would then be using Amazon Athena and Amazon QuickSight to identify patient density in a specific geographic area. This will help them identify potential customers.

![](https://github.com/aws-samples/aws-lakeformation-ml-transforms/raw/master/img/architecture.png)

### [Amazon EMR](https://tutorialsdojo.com/amazon-emr/)

* A managed cluster platform that simplifies running big data frameworks like Apache Hadoop and Apache Spark, on AWS to process and analyze vast amounts of data.
* You can use EMR to transform and move large amounts of data into and out of other AWS data stores and databases

#### Features

* EMR notebooks provide a managed environment, based on Jupyter Notebooks, to help users prepare and visualize data, collaborate with peers, build apps, and perform interactive analysis using EMR clusters.
* You can leverage multiple data stores, including S3, the Hadoop Distributed File System (HDFS), and DynamoDB.



## Machine Learning

* ![](https://docs.aws.amazon.com/images/whitepapers/latest/accenture-ai-scaling-ml-and-deep-learning-models/images/ml-architecture.png)
#### [Link](https://docs.aws.amazon.com/whitepapers/latest/accenture-ai-scaling-ml-and-deep-learning-models/ml-architecture-on-aws.html)

* ![](https://substackcdn.com/image/fetch/$s_!QDEg!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7c644199-33a7-439a-af31-e88548f1f519_1139x883.png)
#### [Link](https://www.dataopslabs.com/p/aws-sagemaker-reference-architecture)

### Amazon Bedrock

* Amazon Bedrock is a fully managed service that provides a unified API to access popular foundation models (FMs). Amazon Bedrock supports image generation models from providers such as Stability AI or AWS.
* You can use Amazon Bedrock to consume FMs through a unified API without the need to train, host, or manage ML models. This is the most suitable solution for a company that does not want to train or manage ML models for image generation.


### Amazon Comprehend

*  Amazon Comprehend is a natural language processing (NLP) service that can extract insights and relationships from text data. You cannot use Amazon Comprehend to process textual information from images that are provided in PNG format. Amazon Comprehend requires text as input.
*  

### Amazon SageMaker


#### SageMaker AutoPilot

#### SageMaker GroundTruth

* SageMaker Ground Truth helps manage human-in-the-loop tasks in your ML lifecycles
* Assign internal teams or commission Mechanical Turk for data labeling tasks
  

#### SageMaker Data Wrangler


#### SageMaker JumpStart

* JumpStart provides pretrained, open-source models for a wide range of problem types to help you get started.
* JumpStart also provides solution templates that set up infrastructure for common use cases, and executable example notebooks for machine learning with SageMaker AI.
* SageMaker JumpStart offers state-of-the-art foundation models for use cases such as content writing, code generation, question answering, copywriting, summarization



#### SageMaker Studio

#### SageMaker Notebooks

#### SageMaker Pipelines

#### SageMaker Model Cards

* Use Amazon SageMaker Model Cards to document critical details about your machine learning (ML) models in a single place for streamlined governance and reporting. Catalog details such as the intended use and risk rating of a model, training details and metrics, evaluation results and observations, and additional call-outs such as considerations, recommendations, and custom info.

#### SageMaker Model Dashboard

* Amazon SageMaker Model Dashboard is a centralized portal, accessible from the SageMaker AI console, where you can view, search, and explore all of the models in your account. You can track which models are deployed for inference and if they are used in batch transform jobs or hosted on endpoints.

#### SageMaker Model Monitor

* Amazon SageMaker Model Monitor monitors the quality of Amazon SageMaker AI machine learning models in production. 
* Set alerts that notify you when there are deviations in the model quality. 
* ![](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/model_monitor/mmv2-architecture.png)


#### SageMaker Role Manager

* Machine Learning (ML) administrators striving for least-privilege permissions with Amazon SageMaker AI must account for diversity of industry perspectives, including the unique least-privilege access needs required for personas such as data scientists, machine learning operation (MLOps) engineers and more. Use Role Manager to build and manage persona-based IAM roles for common machine learning needs directly through the Amazon SageMaker AI console.
* SageMaker Role Manager provides 3 preconfigured role personas and predefined permissions for common ML activities:
  1. Data Scientist Persona
  2. MLOps Persona
  3. SageMaker AI compute Persona

#### [SageMaker Feature Store](https://tutorialsdojo.com/amazon-sagemaker-feature-store/)

* there are three modes that a feature store offers:
  * **Online** - provides low-latecy feature access, making it suitable for high-throughput prediction applications.
  * **Offline** - allows for batch processing of large datasets stored in the offline store. These datasets can be used for training models or performing batch inference. The offline store utilizes S3 for storage and supports data retrieval using Athena queries.
  * **Online and Offline** - a combination of online and offline modes.
* Support both streaming and batch data ingestion.
  * Streaming Ingestion
    * Streaming features allow you to continuously push new or updated feature data to the store in real-time.
    * This is done by using the synchronous `Put Record` API, ensuring the latest feature values are always available.
  * Batch Ingestion
    * Allows you to use tools like SageMaker Data Wrangler to create features and then export a notebook that can be used to ingest the features in batches into a feature group.
    * This method supports both offline and online ingestion, depending on the configuration of the feature group.

#### SageMaker JumpStart

### Amazon Mechanical Turk

* is a crowdsourcing marketplace that connects you with an on-demand, scalable, human workforce to complete tasks.
* This service could be used to help label data

### Amazon Kendra

* Amazon Kendra is an intelligent search service that uses semantic and contextual understanding to provide relevant responses to a search query. You cannot use Amazon Kendra to detect and extract text, handwriting, and data from invoice images.

### Amazon Polly

* Amazon Polly is a text-to-speech (TTS) service that can convert text into lifelike speech. You cannot use Amazon Polly to detect and extract text, handwriting, and data from invoice images.




### Amazon Textract

* Amazon Textract is a service that you can use to add document text detection and analysis to applications. You can use Amazon Textract to identify handwritten text, to extract text from documents, and to extract specific information from documents. Amazon Textract does not provide access to FMs.
* Amazon Textract is fully managed service that can detect and extract text and data from scanned documents, PDFs, and images. One of the use cases for Amazon Textract is to process invoices and receipts. For example, Amazon Textract can detect billing and shipping addresses automatically from images.

## Storage

![](https://docs.aws.amazon.com/images/whitepapers/latest/aws-overview/images/storage-services.png)

![](https://d2908q01vomqb2.cloudfront.net/fc074d501302eb2b93e2554793fcaf50b3bf7291/2023/07/06/Fig1-hybrid-data-access-strategy.png)
##### [Link](https://aws.amazon.com/blogs/architecture/designing-a-hybrid-ai-ml-data-access-strategy-with-amazon-sagemaker/)

### Amazon S3

**Storage Classes**

| Storage Class | Designed for | Durability | Availability | AZs | Min Storage Duration | Other considerations |
| --- | --- | --- | --- | --- | --- | --- |
| S3 Standard (`STANDARD`) | Frequently accessed data (more than once a month), ms access | 11 9s% | 99.99% | >=3 | None | None |
| S3 Standard-IA (`STANDARD_IA`) | Long lived, infrequently accessed (1 a month), ms access | 11 9s% | 99.9% | >=3 | 30 days | Per-GB retrieval fees apply. |
| S3 Intelligent-Tiering | Data with unknown or changing or unpredictable access patterns | 11 9s% | 99.9% | >=3 | None | Monitoring and automation fees per object apply. No retrieval fees |
| S3 One Zone-IA | Recreatable, infrequently accessed data (1 a month) ms access | 11 9s% | 99.5% | 1 | 30 days | per-gb retrieval fees apply. Not resilient to the loss of AZ |
| S3 Express One Zone | Single digit millisecond data access for latency-sensitive applications within a single AWS AZ | 11 9s% | 99.95% | 1 | None | S3 Express one Zone ojbects are stored in a signle AWS AZ that you choose |
| S3 Glacier Instant Retrieval | Long-lived, archive data accessed once a quarter with ms access | 11 9s% | 99.9% | >=3 | 90 days | per-gb retrieval fees apply |
| S3 Glacier Flexible Retrieval | long-lived archival data accessed once a year with retrieval times of minutes to hours |  11 9s% | 99.99% (after you store objects) | >=3| 90 days | Per-GB retrieval fees apply. You must first restore archived objects before you can access them |
| S3 Glacier deep archive | long-lived archive data accessed less than once a year with retrieval times of hours | 11 9s% | 99.99% (after you store the object) | >=3 | 180 days | Per-GB retrieval fees apply. You must first restore acrhived objects before you can access them. |
| Reduced Redundancy Storage (Not recommended) | Noncritical, frequently accessed data with ms access | 99.99% | 99.99% | >=3 | None | None|


**S3 Lifecycle Management**

* Set of rules to move data between different tiers, to save storage cost
* Defining S3 Lifecycle Management can be great for cost savings if your objects have predictable usage patterns.
* Example:
  * Start with S3 Standard with a lifecycle management policy that says if objects have not been accessed for 30 days, move it to S3 Glacier Flexible Retrieval tier. Once in the S3 Glacier, can have another policy that purges after 180 days.
  * S3 Bucket Standard :bucket: :arrow_right:---30days---:arrow_right: S3 Glacier Flex Retrieval :snowflake: :arrow_right:---180days---:arrow_right: :wastebasket:

**S3 Encryption for Objects**

* There are 4 methods of encrypting objects in S3
  1. **SSE-S3**: Encrypts S3 objects using keys handled and managed by AWS.
  2. **SSE-KMS**: us AWS Key Management Service to manage encryption keys
     1. Addtional security (user must have access to KMS key)
     2. Audit trail for KMS key usage
  3. **SSE-C**: when you want to manage your own ecnryption keys
  4. **Client Side Encryption**

**Amazon Lake Formation for granular control**

* AWS Lake formation is a managed service that centrally governs, secures, and manages data access to your data lake stored in Amazon S3. It provides a simplified, centralized, and fine-grained security model that replaces complex S3 bucket policies and IAM policies, enabling secure access for analytics and machine learning workloads.

**S3 Events**

* Automatically trigger event-drive workloads with S3 events
* [Send events to event handlers like SNS, SQS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ways-to-add-notification-config-to-bucket.html), or EventBridge; or directly trigger Lambda functions
* Events can be triggered off of new object creation, object removal, object restoration, and object loss.

**S3 Partitioning**

* Partitioning is a logical organization of data files based on value of more or more fields, commonly **date, region or device id** into a separate folder structures.
* Rather than scan the entire dataset, engines like **Athena, Spark, Hive, or Presto** can prune partitions and read only the relevant data. This speeds up queries and reduces costs.

**S3 Gateway Endpoints**

* You can access Amazon S3 from your VPC using gateway VPC endpoints. After you create the gateway endpoint, you can add it as a target in your route table for traffic destined from your VPC to Amazon S3.
* Amazon S3 supports both gateway endpoints and interface endpoints. With a gateway endpoint, you can access Amazon S3 from your VPC, without requiring an internet gateway or NAT device for your VPC, and with no additional cost.
* However, gateway endpoints do not allow access from on-premises networks, from peered VPCs in other AWS Regions, or through a transit gateway. For those scenarios, you must use an interface endpoint, which is available for an additional cost.
![](https://docs.aws.amazon.com/images/vpc/latest/privatelink/images/gateway-endpoints.png)


### Amazon FSx

* Lustre is an open-source file system designed for HPC environments
* Can scale up to TB/s throughput and millions of IOPS
* FSx Lustre can be directly mounted to EC2 training instances
* FSx is also backed by S3
  * Integrate and sync with S3 to deliver data faster for training
  * leverages S3 for cost-effective cold storage
* FSx can handle hundreds of gigabytes of throughput, and millions of IOPS for super-low-latency file retrieval
* FSx for Lustre has 2 other platforms it supports:
  * **FSx Lustre NetApp ONTAP**
    * Specifically for use with NetApp's ONTAP file system
    * Provides an in-VPC access point to data loaded from an ONTAP server
    * Uses S3 protocol for reads, NFS protocol for Writes
  * **FSx for Windows File Server**
    * Supports SMB protocol used by Windows Servers
    * Only recommended if a shared file system is needed for Windows-based EC2 applications

### Amazon Elastic Block Store (Amazon EBS)

* Block storage that is behind EC2.
* EBS is a scalable storage service purpose-built for use with EC2.
* You can scale the storage independent of the compute.
* Training data can be pre-loaded or streamed to EBS volumes


### Amazon Elastic File System (Amazon EFS)

* EFS is a shared file system that can be mounted directly to Linux EC2 instances or containers for training.
* Can be mounted to multiple instances for parallel processing
* Uses the NFS v4 protocol
* Supports thousands of connections without impacting performance
* Not as cost effective as S3, and not as performant as FSx Lustre
  * Because of cost and performance, generally only recommended for training data if the data already resides in EFS


### AWS Storage Gateway

![](https://d2908q01vomqb2.cloudfront.net/e1822db470e60d090affd0956d743cb0e7cdf113/2020/05/04/Figure-2-High-level-architecture-of-storage-gateway.png)

AWS Storage Gateway is a hybrid cloud storage service that connects on-premises environments with AWS cloud storage. It allows you to seamlessly integrate your existing on-premises architecture with AWS, enabling you to store and retrieve data from the cloud and run applications in a hybrid environment. For Windows workloads, you can use Storage Gateway to store and access data using native Windows protocols like SMB and NFS. You can use storage gateway to reduce costs associated with running Windows workloads on AWS by using on-premises hardware and software as a bridge to the cloud. This enables you to take advantage of the scalability and cost-efficiency of AWS without having to make significant changes to your existing infrastructure.

Under the umbrella of Storage Gateway, you get Amazon S3 File Gateway, Amazon FSx File Gateway, Tape Gateway, and Volume Gateway.

Uses a storage gateway appliances, a VM from Amazon - which is installed and hosted on your data center. After the setup, you can use the AWS console to provision your storage options: File Gateway, Cached Volumes, or Stored Volumns, in which data will be saved to S3. You can also purchase a hardware appliance to facilitate the transfer instead of a VM.


## Migration and Transfer

### AWS DataSync

* Architecture
![](https://docs.aws.amazon.com/images/datasync/latest/userguide/images/DataSync-chart-on-prem.png)
* high-speed file transfer service that helps transfer your file or object data to, from, and between AWS storage services.
* Works with on-prem storage systems:
  * NFS, SMB, HDFS, object storage
* Works with different AWS storage services:
  * S3, EFS, FSx for Windows File Server, FSx for Lustre, FSx for OpenZFS, FSx for NetApp ONTAP
* Uses an agent which is a VM that is owned by the user and is used to read or write data from your storage systems. The agent will then read from a source location, and sync your data to S3, EFS, or FSx for Windows File Server.
* Connecting your network for AWS DataSync
  * ![](https://docs.aws.amazon.com/images/datasync/latest/userguide/images/datasync-network-connection-diagram-overview.png)

### AWS Database Migration Services (AWS DMS)

* high level architecture
  * ![](https://docs.aws.amazon.com/images/dms/latest/userguide/images/datarep-Welcome.png)
* Tool used to migrate relational databases, data warehouses, NoSQL databases, and other types of data stores.









