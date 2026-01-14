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

## Data Storage services in AWS

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

