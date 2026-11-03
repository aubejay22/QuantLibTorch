"""Isolated stock QuantLib date-convention oracle."""
import json
import sys

import QuantLib as ql

assert ql.__version__ == "1.40"
CALENDARS = {"null": ql.NullCalendar, "weekends_only": ql.WeekendsOnly, "target": ql.TARGET}
CONVENTIONS = {"following": ql.Following, "modified_following": ql.ModifiedFollowing,
               "preceding": ql.Preceding, "modified_preceding": ql.ModifiedPreceding,
               "unadjusted": ql.Unadjusted}
UNITS = {"days": ql.Days, "weeks": ql.Weeks, "months": ql.Months, "years": ql.Years}


def evaluate(case):
    value = ql.DateParser.parseISO(case["date"])
    if case["operation"] == "fraction":
        dc = ql.Actual360() if case["day_counter"] == "actual360" else ql.Actual365Fixed()
        return dc.yearFraction(value, ql.DateParser.parseISO(case["end"]))
    cal, convention = CALENDARS[case["calendar"]](), CONVENTIONS[case["convention"]]
    if case["operation"] == "adjust":
        return cal.adjust(value, convention).ISO()
    return cal.advance(value, case["count"], UNITS[case["unit"]], convention,
                       case["end_of_month"]).ISO()


if __name__ == "__main__":
    print(json.dumps([evaluate(case) for case in json.load(sys.stdin)]))
