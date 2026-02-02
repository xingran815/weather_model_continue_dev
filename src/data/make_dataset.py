from datetime import datetime, timedelta
import os
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional


# Definition of the database created with MySQL Docker container
def make_dataset(duration: Optional[int] = 10) -> tuple[str, str]: 
    '''
    Connects to the MySQL database, samples a random subset of 20 % of the weather data,
    and saves it as a CSV file in the specified output directory.
    args:
        duration (int): Duration of the dataset in years.
        default is 10 years.
    Returns:
        OUTPUT_FILE (str): Path to the saved CSV file.
        DATE (str): Timestamp of when the file was created.

    '''
    MYSQL_USER = "root"
    MYSQL_PASSWORD = "root"
    MYSQL_HOST = "weather_sql_container" 
    MYSQL_PORT = 3306
    MYSQL_DB = "weather_db"
    
    
    TABLE_NAME = 'weather_data'
    NEW_TABLE_NAME = 'weather_subset'
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR= os.path.join(THIS_DIR, "../../data/raw") 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DATE = datetime.now().strftime("%Y%m%d_%H%M")
    OUTPUT_FILE = f'{OUTPUT_DIR}/weather_subset_{DATE}.csv'
    
    TABLE_PERCENT = 0.2  # x percent of the data

    engine = create_engine(
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )
    
    #filter query eg.loacation (Canberra, Sydney, Melbourne, Brisbane, Adelaide)
    '''
    all locations:
    'Albury' 'BadgerysCreek' 'Cobar' 'CoffsHarbour' 'Moree' 'Newcastle'
     'NorahHead' 'NorfolkIsland' 'Penrith' 'Richmond' 'Sydney' 'SydneyAirport'
     'WaggaWagga' 'Williamtown' 'Wollongong' 'Canberra' 'Tuggeranong'
     'MountGinini' 'Ballarat' 'Bendigo' 'Sale' 'MelbourneAirport' 'Melbourne'
     'Mildura' 'Nhil' 'Portland' 'Watsonia' 'Dartmoor' 'Brisbane' 'Cairns'
     'GoldCoast' 'Townsville' 'Adelaide' 'MountGambier' 'Nuriootpa' 'Woomera'
     'Albany' 'Witchcliffe' 'PearceRAAF' 'PerthAirport' 'Perth' 'SalmonGums'
     'Walpole' 'Hobart' 'Launceston' 'AliceSprings' 'Darwin' 'Katherine'
     'Uluru']
     '''
    #query = f"SELECT {', '.join(columns_to_load)} FROM {TABLE_NAME} WHERE Location='Sydney'"
    
    
   # Drop the new table if it already exists
    with engine.connect() as conn:
        conn.execute(text(
            f"""
            DROP TABLE IF EXISTS {NEW_TABLE_NAME};
            """
        ))
    
    # Calculate the start and end date of the dataset
    start_date = datetime(year=2008, month=1, day=1)
    end_date = start_date + timedelta(days=365 * duration)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # Filter random x % from the data
    with engine.connect() as conn:
        total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
        sample_size = int(total_rows * TABLE_PERCENT)
        conn.execute(text(
            f"""
            CREATE TABLE {NEW_TABLE_NAME} AS
            SELECT *
            FROM {TABLE_NAME}
            WHERE Date >= '{start_date}' and Date <= '{end_date}'
            ORDER BY RAND()
            LIMIT {sample_size}
            """
        ))
    
    # Validatw the new table creation
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {NEW_TABLE_NAME}")).fetchone()
        print(result)   
    
    
    df = pd.read_sql(f"SELECT * FROM {NEW_TABLE_NAME}", engine)
    
    # Save the sampled data to a CSV file
    df.to_csv(OUTPUT_FILE, index=False)

    return OUTPUT_FILE, DATE
