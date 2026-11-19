
import os
import re
import io
import sqlite3
import pandas as pd


UPLOAD_DIR = "data/uploads"
TABLE_NAME = "dataset"


def clean_column_names(df):

    df = df.copy()

    new_columns = []
    used_columns = set()

    for column in df.columns:

        clean_name = re.sub(
            r"\W+",
            "_",
            str(column).strip()
        ).strip("_")

        if not clean_name:
            clean_name = "column"

        if clean_name[0].isdigit():
            clean_name = f"col_{clean_name}"

        original_name = clean_name
        counter = 1

        while clean_name.lower() in used_columns:
            clean_name = f"{original_name}_{counter}"
            counter += 1

        used_columns.add(clean_name.lower())
        new_columns.append(clean_name)

    df.columns = new_columns

    return df


def save_dataframe(df, session_id, file_name):

    if df.empty:
        raise ValueError("Dataset is empty.")

    df = clean_column_names(df)

    session_folder = os.path.join(
        UPLOAD_DIR,
        session_id
    )

    os.makedirs(session_folder, exist_ok=True)

    db_path = os.path.abspath(
        os.path.join(session_folder, "dataset.db")
    )

    connection = sqlite3.connect(db_path)

    try:

        df.to_sql(
            TABLE_NAME,
            connection,
            if_exists="replace",
            index=False
        )

    finally:

        connection.close()

    return {
        "dataframe": df,
        "db_path": db_path,
        "table_name": TABLE_NAME,
        "columns": df.columns.tolist(),
        "rows": len(df),
        "file_name": file_name
    }


def load_uploaded_dataset(uploaded_file, session_id):

    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".csv"):

        df = pd.read_csv(io.BytesIO(file_bytes))

    elif file_name.endswith((".xlsx", ".xls")):

        df = pd.read_excel(io.BytesIO(file_bytes))

    else:

        raise ValueError(
            "Only CSV and Excel files are supported."
        )

    return save_dataframe(
        df,
        session_id,
        uploaded_file.name
    )
