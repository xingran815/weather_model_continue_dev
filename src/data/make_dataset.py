import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings

logger = logging.getLogger(__name__)


def make_dataset(sample_percent: float | None = 0.2, duration: int | None = 10) -> dict[str, str | None | float | int]:
    """
    Create a sub-dataset from the MySQL weather table and save as CSV.

    Connects to the MySQL database, samples a random subset of the weather data
    within the given duration (years), and saves it as a CSV in data/raw.

    Args:
        sample_percent: Fraction of rows to sample (e.g. 0.2 for 20%). Default 0.2.
        duration: Number of years of data from 2008-01-01. Default 10.

    Returns:
        Dict with keys: raw_data_file, processed_data_file (None), date,
        sample_percent, duration.
    """
    TABLE_NAME = "weather_data"
    NEW_TABLE_NAME = "weather_subset"
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(THIS_DIR, "../../data/raw")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DATE = datetime.now().strftime("%Y%m%d_%H%M")
    OUTPUT_FILE = f"{OUTPUT_DIR}/weather_subset_{DATE}.csv"

    engine = create_engine(
        f"mysql+mysqlconnector://{settings.mysql_user}:{settings.mysql_password}@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_db}"
    )

    # filter query eg.loacation (Canberra, Sydney, Melbourne, Brisbane, Adelaide)
    """
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
     """
    # query = f"SELECT {', '.join(columns_to_load)} FROM {TABLE_NAME} WHERE Location='Sydney'"

    # Drop the new table if it already exists
    with engine.connect() as conn:
        conn.execute(
            text(
                f"""
            DROP TABLE IF EXISTS {NEW_TABLE_NAME};
            """
            )
        )

    # Calculate the start and end date of the dataset
    start_date = datetime(year=2008, month=1, day=1)
    end_date = start_date + timedelta(days=365 * duration)
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    logger.info(
        "Sampling window prepared",
        extra={
            "start_date": start_date,
            "end_date": end_date,
            "sample_percent": sample_percent,
        },
    )

    # Filter random x % from the data (use parameterized query to avoid SQL injection)
    with engine.connect() as conn:
        total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
        sample_size = int(total_rows * sample_percent)
        conn.execute(
            text(
                f"""
                CREATE TABLE {NEW_TABLE_NAME} AS
                SELECT *
                FROM {TABLE_NAME}
                WHERE Date >= :start_date AND Date <= :end_date
                ORDER BY RAND()
                LIMIT :sample_size
                """
            ),
            {"start_date": start_date, "end_date": end_date, "sample_size": sample_size},
        )
        conn.commit()

    # Validate the new table creation
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {NEW_TABLE_NAME}")).fetchone()
        row_count = result[0] if result is not None else 0
        logger.info(
            "Sample table created",
            extra={"table_name": NEW_TABLE_NAME, "row_count": row_count},
        )

    df = pd.read_sql(f"SELECT * FROM {NEW_TABLE_NAME}", engine)

    # Save the sampled data to a CSV file
    df.to_csv(OUTPUT_FILE, index=False)

    # return a json object with the output file and date
    return {
        "raw_data_file": OUTPUT_FILE,
        "processed_data_file": None,
        "date": DATE,
        "sample_percent": sample_percent,
        "duration": duration,
    }
