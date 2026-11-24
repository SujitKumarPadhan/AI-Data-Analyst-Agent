
import pandas as pd
from langchain.tools import tool


def create_dynamic_eda_tools(df):

    dataset = df.copy()

    @tool
    def get_dataset_overview():
        """Show dataset shape, column names, and data types."""

        return {
            "rows": len(dataset),
            "columns_count": len(dataset.columns),
            "columns": dataset.columns.tolist(),
            "data_types": {
                col: str(dtype)
                for col, dtype in dataset.dtypes.items()
            }
        }

    @tool
    def get_missing_values():
        """Find missing values in every dataset column."""

        missing = dataset.isnull().sum()

        return {
            col: int(value)
            for col, value in missing.items()
        }

    @tool
    def get_missing_value_percentage():
        """Calculate missing value count and percentage column-wise."""

        missing_count = dataset.isnull().sum()
        missing_percentage = (
            dataset.isnull().mean() * 100
        ).round(2)

        return {
            col: {
                "missing_count": int(missing_count[col]),
                "missing_percentage": float(missing_percentage[col])
            }
            for col in dataset.columns
        }

    @tool
    def get_duplicate_rows():
        """Detect completely identical duplicate rows."""

        duplicate_count = int(dataset.duplicated().sum())

        return {
            "total_duplicate_rows": duplicate_count,
            "duplicate_percentage": round(
                (duplicate_count / len(dataset)) * 100, 2
            ) if len(dataset) > 0 else 0
        }

    @tool
    def get_duplicate_employee_ids():
        """Detect duplicate Employee_ID values, even when other details are different."""

        if "Employee_ID" not in dataset.columns:
            return {
                "error": "Employee_ID column not found in the dataset."
            }

        employee_ids = dataset["Employee_ID"]

        duplicate_mask = (
            employee_ids.notna()
            & employee_ids.duplicated(keep=False)
        )

        duplicate_records = dataset[duplicate_mask]

        duplicate_ids = {
            str(employee_id): int(count)
            for employee_id, count in
            duplicate_records["Employee_ID"].value_counts().items()
        }

        return {
            "duplicate_employee_id_count": len(duplicate_ids),
            "duplicate_employee_ids": duplicate_ids,
            "affected_rows": int(len(duplicate_records))
        }

    @tool
    def get_numeric_summary():
        """Calculate statistical summary of numeric columns."""

        numeric_df = dataset.select_dtypes(
            include="number"
        )

        if numeric_df.empty:
            return "No numeric columns found."

        return numeric_df.describe().round(2).to_dict()

    @tool
    def get_categorical_summary():
        """Analyze unique values in categorical columns."""

        categorical_df = dataset.select_dtypes(
            include=["object", "category", "bool"]
        )

        if categorical_df.empty:
            return "No categorical columns found."

        result = {}

        for column in categorical_df.columns:
            result[column] = {
                "unique_values": int(
                    categorical_df[column].nunique()
                ),
                "top_values": (
                    categorical_df[column]
                    .value_counts()
                    .head(5)
                    .to_dict()
                )
            }

        return result

    @tool
    def get_correlation():
        """Calculate correlation between numeric columns."""

        numeric_df = dataset.select_dtypes(
            include="number"
        )

        if numeric_df.shape[1] < 2:
            return "At least two numeric columns are required."

        return numeric_df.corr().round(2).to_dict()

    @tool
    def get_top_values(column_name: str):
        """Find the most frequent values in a specified column."""

        if column_name not in dataset.columns:
            return f"Column '{column_name}' does not exist."

        return (
            dataset[column_name]
            .value_counts()
            .head(10)
            .to_dict()
        )

    return [
        get_dataset_overview,
        get_missing_values,
        get_missing_value_percentage,
        get_duplicate_rows,
        get_duplicate_employee_ids,
        get_numeric_summary,
        get_categorical_summary,
        get_correlation,
        get_top_values
    ]
