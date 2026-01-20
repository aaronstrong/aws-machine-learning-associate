# In-Scope AWS Services

### Table of Contents

* Analytics
  * [AWS Lake Formation](#aws-lake-formation)
* Storage
  * [Amazon S3](#amazon-s3)
  * [Amazon FSx](#amazon-fsx)
* Migration Services
  * [AWS DataSync](#aws-datasync)
  * [AWS DMS](#aws-database-migration-services-aws-dms)



## Analytics


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

### Kinesis Data Firehouse

* Deliver to integrated services like S3, Redshift, or Amazon OpenSearch service
* Deliver to popular applications like Splunk, Snowflake, or a custom HTTP endpoint
* Convert data to Parquet or ORC
* Integrate with Lambda for custom transformations
* Dynamically partition data delivered to S3

### Managed Service for Apache Flink

* A way to live process and analyse data as it's streamed
* Interactively query real-time data and generate continuous insights
* Detect outliers and threshold breaches as early as possible

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



## Storage

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



### Amazon Elastic Block Store (Amazon EBS)


### Amazon Elastic File System (Amazon EFS)



## Migration and Transfer

### AWS DataSync

* Architecture
![](https://docs.aws.amazon.com/images/datasync/latest/userguide/images/DataSync-chart-on-prem.png)
* high-speed file transfer service that helps transfer your file or object data to, from, and between AWS storage services.
* Works with on-prem storage systems:
  * NFS, SMB, HDFS, object storage
* Works with different AWS storage services:
  * S3, EFS, FSx for Windows File Server, FSx for Lustre, FSx for OpenZFS, FSx for NetApp ONTAP
* Connecting your network for AWS DataSync
  * ![](https://docs.aws.amazon.com/images/datasync/latest/userguide/images/datasync-network-connection-diagram-overview.png)

### AWS Database Migration Services (AWS DMS)

* high level architecture
  * ![](https://docs.aws.amazon.com/images/dms/latest/userguide/images/datarep-Welcome.png)
* Tool used to migrate relational databases, data warehouses, NoSQL databases, and other types of data stores.


### Kineses Data Streams

* Fully managed service to ingest data streams
* Stream are split into shards
  * Write up to 1 MB or 1000 records per second
  * Read up to 2 MB or 2000 records per second for each shard
  * If Kinesis is under performing, increase the number of shards
* Default limit of 10,000 shards per stream, but there is technically no upper limit





### Amazon Managed Streaming for Apache Kafka (MSK)

* Create Apache Kafka clusters from scratch or deploy your existing Kafka cluster to AWS
* Optimized for capturing log and event streams
* Native integrations iwth Kinesis family, EC2, Lambda, Redshift, and others
* Typically only recommended over Kineses Data Streams in cases where your organization or application is already using Kafka
