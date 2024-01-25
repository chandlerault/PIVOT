import pymssql
import pandas
import yaml

CONFIG_FILE_PATH = 'config/config.yaml'
config = yaml.load(open(CONFIG_FILE_PATH, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

def get_label_rank_df(
    model_id, # int
    dissimilarity_id, # int
    batch_size=100, # float
    relabel_lambda=0.069, # float
    random_ratio=0.5, # float
):
    
    batch_size_d = int((1- random_ratio)*batch_size)
    batch_size_r = batch_size - batch_size_d
    
    # call stored procedure for label ranking w/ D_ID
    args = {
        "M_ID": model_id,
        "D_ID": dissimilarity_id,
        "BATCH_SIZE": batch_size_d,
        "RELABEL_LAMBDA": relabel_lambda,
    }
    d_df = get_data_sp(sp="AL_RANKINGS", args)
    
    # call stored  procedure for label ranking w/ D_ID = 0 represents random
    args['D_ID'] = 0
    args['BATCH_SIZE'] = batch_size_r
    r_df = get_data_sp(sp="AL_RANKINGS", args)
    
    full_df = pd.concat([d_df, r_df]) # might have duplicates but the chance is so low it doesn't matter
    
    return full_df
    

def get_data_sp(
    sp="", # string
    args, # dict
):
    # load from config file
    server = config['server']
    database = config['database']
    username = config['username']
    password = config['password']

    # set up a connection to Azure
    with pymssql.connect(server, user, password, database) as conn:
        # set up a cursor object
        with conn.cursor(as_dict=True) as cursor:
            # gather variables for the stored procedure
            arg_tuples = (args['M_ID'], args['D_ID'], args['RELABEL_LAMBDA'], args['BATCH_SIZE'])
            # execute stored procedure
            cursor.callproc(sp, arg_tuples)
            # Fetch the results into a Pandas DataFrame
            df = pd.DataFrame(cursor.fetchall())
     
    return df
    
    