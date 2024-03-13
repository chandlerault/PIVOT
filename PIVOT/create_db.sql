/*
A set of commands to generate the Azure SQL database.
Tables:
- models: Table to store CNN models
- images: Table to store image filepaths in Azure Blob
- dissimilarity: Table to store dissimilarity and other active learning metrics
- users: Table to store user information
- predictions: Table to store class predictions of images for each model
- metrics: Table to store computed AL metrics per image per model
- labels: Table to store user labels for images
*/

CREATE TABLE models (
    m_id int IDENTITY(1,1) PRIMARY KEY, -- unique model identifier
    model_name VARCHAR(255) NOT NULL, -- name of CNN model
    model_link VARCHAR(MAX), -- MLFlow link to model
    class_map VARCHAR(MAX) NOT NULL -- dictionary of class labels
);

CREATE TABLE images (
    i_id int IDENTITY(1,1) PRIMARY KEY, -- unique image identifier
    filepath VARCHAR(255) NOT NULL UNIQUE -- file path to image in Azure Blob 
);

CREATE TABLE dissimilarity (
    d_id int IDENTITY(1,1) PRIMARY KEY, -- unique dissimilarity identifier
    name VARCHAR(255) NOT NULL, -- name of dissimilarity metric
    formula VARCHAR(MAX) -- formula to calculate dissimilarity metric
);

CREATE TABLE users (
    u_id int IDENTITY(1,1) PRIMARY KEY, -- unique user identifier
    email VARCHAR(255) NOT NULL, -- email of user
    name VARCHAR(255) NOT NULL, -- name of user
    experience INT NOT NULL, -- experience level of user
    lab VARCHAR(255) NOT NULL -- laboratory of user
);

CREATE TABLE predictions (
    m_id int NOT NULL, -- unique model identifier
    i_id int NOT NULL, -- unique image identifier
    class_prob VARCHAR(255)  NOT NULL, -- predicted probability of image belonging to each class
    pred_label VARCHAR(255)  NOT NULL, -- label predicted by model
    PRIMARY KEY (m_id, i_id),
    FOREIGN KEY (m_id) REFERENCES models(m_id),
    FOREIGN KEY (i_id) REFERENCES images(i_id)
);

CREATE TABLE metrics (
    i_id int  NOT NULL, -- unique image identifier
    m_id int NOT NULL, -- unique model identifier
    d_id int NOT NULL, -- unique dissimilarity identifier
    d_value FLOAT NOT NULL, -- value of dissimilarity metric
    PRIMARY KEY (m_id, i_id, d_id),
    FOREIGN KEY (i_id) REFERENCES images(i_id),
    FOREIGN KEY (m_id) REFERENCES models(m_id),
    FOREIGN KEY (d_id) REFERENCES dissimilarity(d_id),
);

CREATE TABLE labels (
    i_id int NOT NULL, -- unique image identifier
    u_id int NOT NULL, -- unique user identifier
    weight int NOT NULL, -- weight of label
    date DATETIME NOT NULL, -- date and time of label creation
    label VARCHAR(255) NOT NULL, -- image label
    FOREIGN KEY (i_id) REFERENCES images(i_id),
    FOREIGN KEY (u_id) REFERENCES users(u_id)
);
