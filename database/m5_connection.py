"""SQL Server connection for the M5 shop database."""

import os

import pyodbc


def connect_m5():
    connection_string = os.environ.get("M5_DB_CONNECTION_STRING")
    if not connection_string:
        server = os.environ.get("M5_SQL_SERVER", r".\SQLEXPRESS")
        connection_string = (
            "DRIVER={ODBC Driver 18 for SQL Server};"
            f"SERVER={server};DATABASE=DynamicPricingM5;"
            "Trusted_Connection=yes;TrustServerCertificate=yes;"
        )
    return pyodbc.connect(connection_string, timeout=5)
