"""Domain-specific tools for the Data Analysis Specialist."""

from typing import Any, Dict, List
import statistics
from langchain_core.tools import tool


@tool
def calculate_statistics(data: List[float]) -> Dict[str, Any]:
    """Compute comprehensive statistical metrics (mean, median, stdev, min, max, sum) for a numerical dataset.

    Args:
        data: List of numerical values to analyze.
    """
    if not data:
        return {"error": "Dataset is empty"}

    return {
        "count": len(data),
        "mean": round(statistics.mean(data), 4),
        "median": round(statistics.median(data), 4),
        "stdev": round(statistics.stdev(data), 4) if len(data) > 1 else 0.0,
        "min": min(data),
        "max": max(data),
        "sum": round(sum(data), 4),
    }


@tool
def query_dataset(records: List[Dict[str, Any]], filter_key: str, filter_value: Any) -> List[Dict[str, Any]]:
    """Query and filter a dataset of structured records matching a key-value condition.

    Args:
        records: List of dictionaries representing table rows.
        filter_key: Column or attribute name to filter by.
        filter_value: Value to match.
    """
    return [r for r in records if r.get(filter_key) == filter_value]


@tool
def aggregate_data(records: List[Dict[str, Any]], group_by: str, agg_key: str) -> Dict[str, float]:
    """Group tabular records by a key and calculate total sums for an aggregation column.

    Args:
        records: List of dictionaries representing rows.
        group_by: Categorical column to group records by.
        agg_key: Numerical column to aggregate.
    """
    totals: Dict[str, float] = {}
    for r in records:
        group = str(r.get(group_by, "unknown"))
        val = float(r.get(agg_key, 0.0))
        totals[group] = round(totals.get(group, 0.0) + val, 4)
    return totals


def get_data_tools():
    """Returns the standard tool suite for Data Analysis specialists."""
    return [calculate_statistics, query_dataset, aggregate_data]

