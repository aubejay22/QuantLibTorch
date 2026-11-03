// Passive financial conventions, delegated directly to pinned QuantLib.
Date convention_date(int64_t serial) {
    TORCH_CHECK(serial >= Date::minDate().serialNumber() && serial <= Date::maxDate().serialNumber(),
                "date is outside QuantLib's supported range");
    return Date(static_cast<Date::serial_type>(serial));
}

Calendar selected_calendar(int64_t kind) {
    switch (kind) {
      case 0: return NullCalendar();
      case 1: return WeekendsOnly();
      case 2: return TARGET();
      default: TORCH_CHECK(false, "unsupported calendar");
    }
}

DayCounter selected_day_counter(int64_t kind) {
    switch (kind) {
      case 0: return Actual365Fixed();
      case 1: return Actual360();
      default: TORCH_CHECK(false, "unsupported day counter");
    }
}

BusinessDayConvention selected_convention(int64_t kind) {
    switch (kind) {
      case 0: return Following;
      case 1: return ModifiedFollowing;
      case 2: return Preceding;
      case 3: return ModifiedPreceding;
      case 4: return Unadjusted;
      default: TORCH_CHECK(false, "unsupported business-day convention");
    }
}

int64_t date_adjust(int64_t serial, int64_t calendar, int64_t convention) {
    return selected_calendar(calendar).adjust(convention_date(serial), selected_convention(convention)).serialNumber();
}

int64_t date_advance(int64_t serial, int64_t count, int64_t unit,
    int64_t calendar, int64_t convention, bool end_of_month) {
    TORCH_CHECK(unit >= 0 && unit <= 3, "unsupported period unit");
    TORCH_CHECK(count >= std::numeric_limits<int>::min() && count <= std::numeric_limits<int>::max(),
                "period count exceeds QuantLib integer range");
    const TimeUnit units[] = {Days, Weeks, Months, Years};
    return selected_calendar(calendar).advance(convention_date(serial), static_cast<Integer>(count), units[unit],
        selected_convention(convention), end_of_month).serialNumber();
}

double year_fraction(int64_t start, int64_t end, int64_t day_counter) {
    return selected_day_counter(day_counter).yearFraction(convention_date(start), convention_date(end)).value();
}
