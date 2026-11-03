"""Original QuantLib calendar and day-count operations for fixed configuration."""
from __future__ import annotations

from datetime import date, timedelta
import numbers

import torch

from .fd import _serial


_CALENDARS = {"null": 0, "weekends_only": 1, "target": 2}
_CONVENTIONS = {"following": 0, "modified_following": 1, "preceding": 2,
                "modified_preceding": 3, "unadjusted": 4}
_DAY_COUNTERS = {"actual365fixed": 0, "actual360": 1}


def _choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}")
    return choices[value]


def _date(serial: int) -> date:
    return date(1899, 12, 31) + timedelta(days=serial - 1)


def adjust_date(value: date, *, calendar: str = "weekends_only",
                convention: str = "following") -> date:
    """Adjust a date using QuantLib NullCalendar, WeekendsOnly or TARGET.

    Supported conventions are following, modified_following, preceding,
    modified_preceding and unadjusted. Pricing APIs consume exact dates;
    adjustment is an explicit configuration step, never implicit in pricing.
    """
    return _date(torch.ops.quantlibtorch.date_adjust(
        _serial("value", value), _choice("calendar", calendar, _CALENDARS),
        _choice("convention", convention, _CONVENTIONS)))


def advance_date(value: date, count: int, unit: str = "days", *,
                 calendar: str = "weekends_only", convention: str = "following",
                 end_of_month: bool = False) -> date:
    """Advance a date with the original QuantLib Calendar.advance rules.

    ``days`` counts business days for non-null calendars. Weeks, months and
    years advance calendar periods before business-day adjustment. Negative
    counts are allowed. End-of-month handling applies under QuantLib's rules.
    """
    if isinstance(count, bool) or not isinstance(count, numbers.Integral):
        raise TypeError("count must be an integer")
    if not isinstance(end_of_month, bool):
        raise TypeError("end_of_month must be a bool")
    units = {"days": 0, "weeks": 1, "months": 2, "years": 3}
    return _date(torch.ops.quantlibtorch.date_advance(
        _serial("value", value), int(count), _choice("unit", unit, units),
        _choice("calendar", calendar, _CALENDARS),
        _choice("convention", convention, _CONVENTIONS), end_of_month))


def year_fraction(start: date, end: date, *, day_counter: str = "actual365fixed") -> float:
    """Original Actual/365 Fixed or Actual/360 year fraction between fixed dates.

    The same names select curve/process day counters in Heston and shout
    pricing and mesh preparation. Other existing engines retain their current
    Actual/365 Fixed convention.
    """
    return torch.ops.quantlibtorch.year_fraction(
        _serial("start", start), _serial("end", end),
        _choice("day_counter", day_counter, _DAY_COUNTERS))
