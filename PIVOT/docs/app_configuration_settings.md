## Blob Storage

This application was designed to use Azure Blob Storage to host the images used for model evaluation and validation. You can find how to setup blob storage and more details in the [Azure documentation](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction).

## Azure Database Setup

This application was setup and developed with an Azure SQL database in mind. For information regarding the setup of such a database, please see the [Azure documentation](https://learn.microsoft.com/en-us/azure/azure-sql/database/single-database-create-quickstart?view=azuresql&tabs=azure-portal). Once you setup the resource, you will need to ensure that the database can be accessed from user name and password rather than by Entra ID. You can do this in the settings tab of the server resource you created.

## SQL Database

Once you have your database resource setup, you can create the SQL database structure for the application. Run the queries found in create_db.sql to create the necessary tables for the application. Once the tables are in place, you can begin data ingestion. A template for how to do this is found in [PIVOT/notebooks/initial_data_ingestion.ipynb](https://github.com/chandlerault/PIVOT/blob/main/notebooks/Initital_Data_Ingestion.ipynb).

## Azure ML Model

[Azure Machine Learning SDK for Python Documentation](https://learn.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

#### Subscription ID

Blah blahd bladhhw dewhud wlink

#### Resource Group

#### Workspace Name

#### Experiment Name

#### Model API Key

#### Model Name

#### Model Endpoint Name

#### Model Deployment Name
