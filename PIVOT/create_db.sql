CREATE TABLE models (
    m_id int IDENTITY(1,1) PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_link VARCHAR(MAX),
    class_map VARCHAR(MAX) NOT NULL
);

CREATE TABLE images (
    i_id int IDENTITY(1,1) PRIMARY KEY,
    filepath VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE dissimilarity (
    d_id  int IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    formula VARCHAR(MAX)
);

CREATE TABLE users (
    u_id  int IDENTITY(1,1) PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    experience INT NOT NULL, 
    lab VARCHAR(255) NOT NULL
);

CREATE TABLE predictions (
    m_id int NOT NULL,
    i_id int NOT NULL,
    class_prob VARCHAR(255)  NOT NULL,
    pred_label VARCHAR(255)  NOT NULL,
    PRIMARY KEY (m_id, i_id),
    FOREIGN KEY (m_id) REFERENCES models(m_id),
    FOREIGN KEY (i_id) REFERENCES images(i_id)
);

CREATE TABLE metrics (
    i_id int  NOT NULL,
    m_id int NOT NULL,
    d_id int NOT NULL,
    d_value FLOAT NOT NULL,
    PRIMARY KEY (m_id, i_id, d_id),
    FOREIGN KEY (i_id) REFERENCES images(i_id),
    FOREIGN KEY (m_id) REFERENCES models(m_id),
    FOREIGN KEY (d_id) REFERENCES dissimilarity(d_id),
);

CREATE TABLE labels (
    i_id int NOT NULL,
    u_id int NOT NULL,
    weight int NOT NULL,
    date DATETIME NOT NULL,
    label VARCHAR(255) NOT NULL,
    FOREIGN KEY (i_id) REFERENCES images(i_id),
    FOREIGN KEY (u_id) REFERENCES users(u_id)
);
