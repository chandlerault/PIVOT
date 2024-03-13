## Blob Storage

This application was designed to use Azure Blob Storage to host the images used for model evaluation and validation. You can find how to setup blob storage and more details in the [Azure documentation](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-blobs-introduction).

## Azure Database Setup

This application was setup and developed with an Azure SQL database in mind. For information regarding the setup of such a database, please see the [Azure documentation](https://learn.microsoft.com/en-us/azure/azure-sql/database/single-database-create-quickstart?view=azuresql&tabs=azure-portal). Once you setup the resource, you will need to ensure that the database can be accessed from user name and password rather than by Entra ID. You can do this in the settings tab of the server resource you created.

## SQL Database

Once you have your database resource setup, you can create the SQL database structure for the application. Run the queries found in [create_db.sql](../PIVOT/create_db.sql) to create the necessary tables for the application. Once the tables are in place, you can begin data ingestion. A template for how to do this is found in [PIVOT/notebooks/Initial_Data_Ingestion.ipynb](../notebooks/Initital_Data_Ingestion.ipynb).

## Azure ML Model

We use [MLFlow](https://mlflow.org/docs/latest/models.html) to track and register our model, as well as store model weights and artifacts. Our model is then deployed on the Azure Machine Learning Studio via an [online endpoint](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=azure-cli) which allows for real-time inference via a scalable REST endpoint. A custom scoring script can be used for [deployed MLFlow models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models-online-endpoints?view=azureml-api-2&tabs=cli).

Learn more about hosting models on Azure ML via Python scripts with the [Azure Machine Learning SDK for Python Documentation](https://learn.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py).
