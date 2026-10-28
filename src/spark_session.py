from __future__ import annotations


def get_spark():
    try:
        session = spark  # type: ignore[name-defined]
    except NameError:
        from databricks.connect import DatabricksSession

        session = DatabricksSession.builder.getOrCreate()
    session.conf.set("spark.driver.maxResultSize", "8g")
    return session
