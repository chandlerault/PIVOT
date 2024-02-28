CREATE TABLE models (
    m_id int IDENTITY(1,1) PRIMARY KEY, -- Unique model identifier
    model_name VARCHAR(255) NOT NULL, -- Name and version of CNN model
    model_link VARCHAR(MAX), -- Model MLFlow tracking URI
    class_map VARCHAR(MAX) NOT NULL -- Dictionary of class labels
);

CREATE TABLE images (
    i_id int IDENTITY(1,1) PRIMARY KEY, -- Unique image identifier
    filepath VARCHAR(255) NOT NULL UNIQUE -- Filepath to image stored in Azure Blob
);

CREATE TABLE dissimilarity (
    d_id int IDENTITY(1,1) PRIMARY KEY, -- Unique dissimilarity identifier
    name VARCHAR(255) NOT NULL, -- Name of dissimilarity metrics
    formula VARCHAR(MAX) -- Formula for calculating dissimilarity
);

CREATE TABLE users (
    u_id int IDENTITY(1,1) PRIMARY KEY, -- Unique user identifier
    email VARCHAR(255) NOT NULL, -- Email address of user
    name VARCHAR(255) NOT NULL, -- Name of user
    experience INT NOT NULL, -- Experience level of user
    lab VARCHAR(255) NOT NULL -- User's laboratory
);

CREATE TABLE predictions (
    m_id int NOT NULL, -- Unique model identifier
    i_id int NOT NULL, -- Unique image identifier
    class_prob VARCHAR(255)  NOT NULL, -- Probability of image belonging to class
    pred_label VARCHAR(255)  NOT NULL, -- Predicted class label for image
    PRIMARY KEY (m_id, i_id),
    FOREIGN KEY (m_id) REFERENCES models(m_id),
    FOREIGN KEY (i_id) REFERENCES images(i_id)
);

CREATE TABLE metrics (
    i_id int NOT NULL, -- Unique image identifier
    m_id int NOT NULL, -- Unique model identifier
    d_id int NOT NULL, -- Unique dissimilarity identifier
    d_value FLOAT NOT NULL, -- Value of dissimilarity metric
    PRIMARY KEY (m_id, i_id, d_id),
    FOREIGN KEY (i_id) REFERENCES images(i_id),
    FOREIGN KEY (m_id) REFERENCES models(m_id),
    FOREIGN KEY (d_id) REFERENCES dissimilarity(d_id),
);

CREATE TABLE labels (
    i_id int NOT NULL, -- Unique image identifier
    u_id int NOT NULL, -- Unique user identifier
    weight int NOT NULL, -- Weight of label
    date DATETIME NOT NULL, -- Date that label was created
    label VARCHAR(255) NOT NULL, -- Label of image
    FOREIGN KEY (i_id) REFERENCES images(i_id),
    FOREIGN KEY (u_id) REFERENCES users(u_id)
);
