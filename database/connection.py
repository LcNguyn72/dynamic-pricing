"""Connection settings shared by the importer and API."""

import os

import pyodbc


def connect():
    connection_string = os.environ.get("PRICING_DB_CONNECTION_STRING")
    if not connection_string:
        server = os.environ.get("PRICING_SQL_SERVER", r".\SQLEXPRESS")
        database = os.environ.get("PRICING_DB_NAME", "DynamicPricingDemo")
        connection_string = (
            "DRIVER={ODBC Driver 18 for SQL Server};"
            f"SERVER={server};DATABASE={database};"
            "Trusted_Connection=yes;TrustServerCertificate=yes;"
        )
    return pyodbc.connect(connection_string, timeout=5)
