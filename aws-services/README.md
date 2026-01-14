# In-Scope AWS Services

### Table of Contents

* Analytics
  * [AWS Lake Formation](#aws-lake-formation)
* Storage
  * [Amazon S3](#amazon-s3)
  * [Amazon FSx](#amazon-fsx)



## Analytics

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

| Feature | S3 Standard | S3 Standard-IA | S3 Express One Zone | S3 One Zone-IA | Glacier Instance Retrieval | Glacier Flexible Retrieval | Glacier Deep Archive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Availability |
| Durability |
| Min. storage duration |
| Retrieval Time |
| Availability Zones | Multiple | Multiple | Single



**S3 Lifecycle Management**

* Defining S3 Lifecycle Management can be great for cost savings if your objects have predictable usage patterns.
* Example:
  * Start with S3 Standard with a lifecycle management policy that says if objects have not been accessed for 30 days, move it to S3 Glacier Flexible Retrieval tier. Once in the S3 Glacier, can have another policy that purges after 180 days.
  * S3 Bucket :bucket: :arrow_right: :snowflake: :wastebasket:

**Amazon Lake Formation for granular control**

### Amazon FSx



### Amazon Elastic Block Store (Amazon EBS)


### Amazon Elastic File System (Amazon EFS)