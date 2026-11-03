"""Original calendars, business-day rules, end-of-month and day counters."""
from datetime import date
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

from quantlibtorch import adjust_date, advance_date, year_fraction


def stock(cases):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("stock_conventions_reference.py"))],
                            input=json.dumps(cases), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_original_calendar_adjustments_at_holidays_and_month_end():
    cases = [dict(operation="adjust", date=d, calendar=cal, convention=conv)
             for d, cal, conv in itertools.product(
                 ["2025-08-31", "2025-04-18", "2025-04-21", "2025-12-25", "2025-01-01"],
                 ["null", "weekends_only", "target"],
                 ["following", "modified_following", "preceding", "modified_preceding", "unadjusted"])]
    expected = stock(cases)
    actual = [adjust_date(date.fromisoformat(c["date"]), calendar=c["calendar"],
                           convention=c["convention"]).isoformat() for c in cases]
    assert actual == expected


def test_original_period_advance_business_days_and_end_of_month():
    cases = [dict(operation="advance", date=d, count=count, unit=unit, calendar=cal,
                  convention="modified_following", end_of_month=eom)
             for d, count, unit, cal, eom in itertools.product(
                 ["2024-02-29", "2025-01-31", "2025-04-17"], [-2, 0, 2],
                 ["days", "weeks", "months", "years"], ["null", "weekends_only", "target"], [False, True])]
    expected = stock(cases)
    actual = [advance_date(date.fromisoformat(c["date"]), c["count"], c["unit"],
                          calendar=c["calendar"], convention=c["convention"],
                          end_of_month=c["end_of_month"]).isoformat() for c in cases]
    assert actual == expected


@pytest.mark.parametrize("day_counter", ["actual365fixed", "actual360"])
def test_original_day_counter_leap_negative_and_zero_periods(day_counter):
    cases = [dict(operation="fraction", date=a, end=b, day_counter=day_counter)
             for a, b in [("2024-01-01", "2025-01-01"), ("2026-02-17", "2025-03-11"),
                          ("2025-01-01", "2025-01-01")]]
    assert [year_fraction(date.fromisoformat(c["date"]), date.fromisoformat(c["end"]),
                          day_counter=day_counter) for c in cases] == stock(cases)


@pytest.mark.parametrize("operation,kwargs", [
    (adjust_date, {"calendar": "missing"}), (adjust_date, {"convention": "missing"}),
    (advance_date, {"count": 1.5}), (advance_date, {"count": 1, "unit": "missing"}),
    (advance_date, {"count": 1, "end_of_month": 1}),
])
def test_invalid_conventions_fail_explicitly(operation, kwargs):
    with pytest.raises((TypeError, ValueError)):
        operation(date(2025, 1, 1), **kwargs)
