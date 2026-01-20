# Domain 1: Data Preparation for Machine Learning (ML) - 28%

## Tasks
* Task 1.1: Ingest and store data
* Task 1.2: Transform data and perform feature engineering
* Task 1.3: Ensure data integrity and prepare data for modeling

---

## Task 1.1 Ingest and store data

Knowledge of:
* Data formats and ingestion mechanisms (for example, validated and non-validated formats, Apache Parquet, JSON, CSV, Apache ORC, Apache Avro, RecordIO)
* How to use the core AWS data sources (for example, Amazon S3, Amazon Elastic File System [Amazon EFS], Amazon FSx for NetApp ONTAP)
* How to use AWS streaming data sources to ingest data (for example, Amazon Kinesis, Apache Flink, Apache Kafka)
* AWS storage options, including use cases and tradeoffs

Skills in:
* Extracting data from storage (for example, Amazon S3, Amazon Elastic Block Store [Amazon EBS], Amazon EFS, Amazon RDS, Amazon DynamoDB) by using relevant AWS service options (for example, Amazon S3 Transfer Acceleration, Amazon EBS Provisioned IOPS)
* Choosing appropriate data formats (for example, Parquet, JSON, CSV, ORC) based on data access patterns Ingesting data into Amazon SageMaker Data Wrangler and SageMaker Feature Store
* Merging data from multiple sources (for example, by using programming techniques, AWS Glue, Apache Spark)
* Troubleshooting and debugging data ingestion and storage issues that involve capacity and scalability
* Making initial storage decisions based on cost, performance, and data structure

## Task 1.2: Transform data and perform feature engineering

Knowledge of:
* Data cleaning and transformation techniques (for example, detecting and treating outliers, imputing missing data, combining, deduplication)
* Feature engineering techniques (for example, data scaling and standardization, feature splitting, binning, log transformation, normalization)
* Encoding techniques (for example, one-hot encoding, binary encoding, label encoding, tokenization)
* Tools to explore, visualize, or transform data and features (for example, SageMaker Data Wrangler, AWS Glue, AWS Glue DataBrew)
* Services that transform streaming data (for example, AWS Lambda, Spark)
* Data annotation and labeling services that create high-quality labeled datasets

Skills in:
* Transforming data by using AWS tools (for example, AWS Glue, DataBrew, Spark running on Amazon EMR, SageMaker Data Wrangler)
* Creating and managing features by using AWS tools (for example, SageMaker Feature Store)
* Validating and labeling data by using AWS services (for example, SageMaker Ground Truth, Amazon Mechanical Turk)

## Task 1.3: Ensure data integrity and prepare data for modeling

Knowledge of:
* Pre-training bias metrics for numeric, text, and image data (for example, class imbalance [CI], difference in proportions of labels [DPL])
* Strategies to address CI in numeric, text, and image datasets (for example, synthetic data generation, resampling)
* Techniques to encrypt data
* Data classification, anonymization, and masking
* Implications of compliance requirements (for example, personally identifiable information [PII], protected health information [PHI], data residency)

Skills in:
* Validating data quality (for example, by using DataBrew and AWS Glue Data Quality)
* Identifying and mitigating sources of bias in data (for example, selection bias, measurement bias) by using AWS tools (for example, SageMaker Clarify)
* Preparing data to reduce prediction bias (for example, by using dataset splitting, shuffling, and augmentation)
* Configuring data to load into the model training resource (for example, Amazon EFS, Amazon FSx)

----

# My Notes

---
# Data for Machine Learning Workloads

* Tabular data
* Image Data
* Text Data
* Time Series Data

## Data Formats

* Structured Data
* Semi-structured data
* Unstructured Data

### Data File Types: Structured Data

* Row-Based: Protobut or Avro recordIO
* Column-based: Apache Parquet, ORC (Hive or Spike)

### Semi-structure Data

* Row-based: Microsoft Excel, CSV
* Object-Notation: JSON, JSONL
  * Example: JSON
    ```json
    [
        {
            "uniqueID": 111222,
            "name": "Bob, Billy",
            "age": 57,
        }
    ]
    ```

### Unstructure Data

* Images: .png, .jpg
* Video: .mp4, ogg, .webm
* text: .txt

## What makes Good Data?

* Quantity
  * Must have enough data to train and evalate your ML model
* Quality
  * Must have the 5 R's:
    * Relevent
    * Representative
    * Rich
    * Reliable
    * Responsible

## Data Storage Services in AWS

* Data Lake
  * AWS S3 and [AWS Lake Formation](../aws-services/README.md)
  * Data lakes can store structured, semi-structured, or unstructured data
* Data Warehouse
  * Amazon Redshift
    * Redshift is a data warehouse that can store structure data optimized for business analytics
* Relational Database
  * AWS RDS
    * Managed Relational database service
    * Stores structured data in popular DB engines
    * Used for:
      * OLTP
      * Web Applications
      * Mobile Application
      * PostgreSQL can be used for vector search
* Non-relational Database
  * Amazon DynamoDB
    * Serverless NoSQL database
    * Store semi-structured data in JSON-like items
    * Single-digit millisecond latency at any scale
    * Facilitates event-driven architectures with DynamoDB Streams

## Dat Storage Services on AWS: Storage for ML

* Amazon S3
  * Most flexible of all the services
  * Used as a data lake, intermediate data storage, and for training and evaluation data
* Amazon Elastic Block Store (EBS)
  * Block storage that is behind EC2.
  * EBS is a scalable storage service purpose-built for use with EC2.
  * You can scale the storage independent of the compute.
  * Training data can be pre-loaded or streamed to EBS volumes
* Amazon Elastic File System (EFS)
  * EFS is a shared file system that can be mounted directly to Linux EC2 instance for training.
  * Can be mounted to multiple instances for parallel processing
* Amazon FSx for Lustre
  * FSx Lustre can be directly mounted to EC2 training instances
  * FSx is also backed by S3
  * FSx can handle hundreds of gigabytes of throughput, and millions of IOPS for super-low-latency file retrieval

### Deeper into S3

* Storage Classes
  * Know there are different storage classes for access levels, locations, and costs
* S3 Lifecycle Management
  * S3 lifecycle management is used to help reduce costs. Lifecycle management enabled, you can move objects between different S3 storage class tiers.
  * Configure rules to move objects and even purge data.
* AWS Lake formation
  * AWS Lake formation is a managed service that centrally governs, secures, and manages data access to your data lake stored in Amazon S3. It provides a simplified, centralized, and fine-grained security model that replaces complex S3 bucket policies and IAM policies, enabling secure access for analytics and machine learning workloads.
* S3 Events
  * Automatically trigger event-drive workloads with S3 events
  * IE: Trigger an event when a new .CSV file has been uploaded to an S3 bucket. Send the alert to EventBridge which can then trigger a Lambda function to validate or clean the data.
* S3 Partitioning
  * organize data within Amazon S3 into a hierarchical folder structure based on teh values of one or more objects metadata fields (eg: date, customer id, region).
  * Crucial for optimizing the performance and reducing the cost of big data analytics queries by limiting the amount of data scanned.
* S3 Gateway Endpoints
  * Allows EC2 instances to access S3 bucket and objects without having to go over the public internet.
  * A Gateway Endpoint is created in the Amazon VPC.
  * This the cheapest, fastest, and more secure method to access a bucket.


### Data Ingestion: Batch vs Streaming

There are two ways to ingest data:
1. Batch Ingestion
2. Streaming Ingestion

**Batch Ingestion**

* This data arrives in batches, often one-time or on a recurring schedule. Used for historical data analysis and ML model training
* Use tools like AWS DataSync to move files and objects
* Use tools like AWS DMS to copy relational databases

**Streaming Ingestion**
* This data is continuously generated, consumed, and processed. Used for real-time analysis and ML inference streaming
* Use AWS tools like Kineses Data Streams to stream data from mobile devices, IoT devices, live gaming data, etc, to an S3 bucket.
  * For live processing and insights, use Amazon Managed Service for Apache Flink
  * To deliver the data to different AWS services, use Kinesis Data Firehouse 


---

## Ingest and Store Data

