from __future__ import annotations


def get_spark():
    try:
        return spark  # type: ignore[name-defined]
    except NameError:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()
