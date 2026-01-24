# Domain 1: Data Preparation for Machine Learning (ML) - 28%

## Tasks
* Task 1.1: Ingest and store data
* Task 1.2: Transform data and perform feature engineering
* Task 1.3: Ensure data integrity and prepare data for modeling

---

## Task 1.1 Ingest and store data

Knowledge of:
* [Data formats](#data-formats) and [ingestion mechanisms](#data-ingestion-batch-vs-streaming) (for example, validated and non-validated formats, Apache Parquet, JSON, CSV, Apache ORC, Apache Avro, RecordIO)
* How to use the core [AWS data sources](#data-storage-services-in-aws) (for example, Amazon S3, Amazon Elastic File System [Amazon EFS], Amazon FSx for NetApp ONTAP)
* How to use AWS streaming data sources to ingest data (for example, [Amazon Kinesis](../aws-services/README.md#amazon-kinesis), [Apache Flink](../aws-services/README.md#managed-service-for-apache-flink), Apache Kafka)
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

### Unstructured Data

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

* **Amazon S3**
  * Most flexible of all the services
  * Used as a data lake, intermediate data storage, and for training and evaluation data
* **Amazon Elastic Block Store (EBS)**
  * Block storage that is behind EC2.
  * EBS is a scalable storage service purpose-built for use with EC2.
  * You can scale the storage independent of the compute.
  * Training data can be pre-loaded or streamed to EBS volumes
* **Amazon Elastic File System (EFS)**
  * EFS is a shared file system that can be mounted directly to Linux EC2 instances or containers for training.
  * Can be mounted to multiple instances for parallel processing
  * Uses the NFS v4 protocol
  * Supports thousands of connections without impacting performance
  * Not as cost effective as S3, and not as performant as FSx Lustre
    * Because of cost and performance, generally only recommended for training data if the data already resides in EFS
* **Amazon FSx for Lustre**
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
* To keep Batch Ingestion Scalable:
  * Automate data preprocessing with Step Functions or SageMaker Data Pipelines
  * Leverage scalable managed services for processing like AWS Glue and SageMaker Data Wrangler
  * Use Scheduled scaling (AWS Step Functions) for preprocessing steps that place on SageMaker or EMR
    * AWS Step Function:
      * A service to orchestrate event-driven steps from different Amazon Services
        ![](https://docs.aws.amazon.com/images/step-functions/latest/dg/images/step-functions-example.png)

**Streaming Ingestion**
* This data is continuously generated, consumed, and processed. Used for real-time analysis and ML inference streaming
* Use AWS tools like Kineses Data Streams to stream data from mobile devices, IoT devices, live gaming data, etc, to an S3 bucket.
  * For live processing and insights, use Amazon Managed Service for Apache Flink
  * To deliver the data to different AWS services, use Kinesis Data Firehouse
  * To keep streaming ingestion scalable:
    * Use `JSONL` file format to efficiently stream data with diverse structures
    * If your Kinesis throughput is the bottleneck, increase the number of shards in your Kinesis stream
    * Partition data dynamically as it is delivered to S3 using Kinesis Data Firehose


---

## Ingest and Store Data

### Merging data from multiple sources

| High operational overhead<br>Highly customizable<br>Code Heavy | <--- | ---> | Low Operational overhead<br>Less customizable<br>Code-light |
| --- |  --- |  --- |  --- |
| Amazon EMR | AWS Glue | Amazon SageMaker<br>Data Wrangler | AWS Glue DataBrew |
| ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT2QGO_tfSoRlZkkaBR-X61NgrdRmped6IMmQ&s) | ![](https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/1/aws-glue-9ztw380gkkd1g54iwwsq7.png/aws-glue-g9i4j0s3igbjmai4vernz9.png?_a=DATAg1AAZAA0) | ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQLoa3Zj2aWc5rGAMIDJ73wTDJBCGS56TMmRg&s) | ![](https://miro.medium.com/0*0ruq4bjF8zFGlV7f.jpeg) |

**Amazon EMR**

* A managed Hadoop cluster for running big data operations
  * Hadoop an open-source data framework from Apache
    * Data Preparation (ETL) - Use either Apache Spark or Hive
    * Data Analysis - Use Spark MLlib or presto
* Allows for simplified ETL for large amounts of data into and out of AWS data stores

**ETL with AWS Glue**

* AWS Glue Studio you can import from an S3 bucket, and then create Glue Data Catalogs
* From the catalog, apply transformations to the data, and then export to another S3 bucket.
* After data transformations have been applied and exported to a bucket, other applications like SageMaker (or Athena like diagram) can be used to read that second S3 bucket.
  ![](https://d2908q01vomqb2.cloudfront.net/b6692ea5df920cad691c20319a6fffd7a4a766b8/2024/11/26/image1-7.jpg)

**Amazon EMR vs AWS Glue**

* Similarities
  * Both solutions use `Apache Spark`
* Differences
  * Amazon EMR
    * Amazon EMR has superior price-performance
    * Uses the open-source ecosystem. Lots of plugins to use: Hive, presto, trino, hadoop
  * Glue
    * AWS Glue has superior operation efficiency
    * More built-in features for data discovery, connectors, job monitoring, and orchestration


**AWS Glue Databrew**

* Is a visual data preparation tool that enables users to clean and normalize data without having to write any code. There are over 250 ready-made transformations to automate data preparation tasks, like filtering anomalies, converting data to standard formats, and correcting invalid values.
* Define and reuse transformations
* Here are some common data transformations used:
  * Remove or replace missing values
  * Combine datasets
  * Create columns
  * Filter Data
  * Label Mapping
  * Aggregate Data

**Amazon SageMaker Data Wrangler vs Glue Databrew**

| | Glue Databrew | SageMaker Data Wrangler |
| --- | --- | --- |
| Processing resources | serverless | serverless |
| Visaulizations (built-in) | yes | yes |
| Built-in Transformations | yes (250) | yes |
| Custom transforms /<br>custom code | no | yes (Pandas, SQL, PySpark) |
| Bias detection integration | no | yes (SageMaker Clarify) |
| Feature Store Integration | no | yes (SageMaker Feature Store) |
| Pipeline (CI/CD) | no | yes (SageMaker Pipelines) |


## Data Transformation

