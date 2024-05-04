from datetime import datetime

import calendar

# CONST FORMATDATETODAY= "%d"


def formatTimestampToDay(timestamp):
    return datetime.fromtimestamp(timestamp).day

def convertMonthtoLatin(month):
    return calendar.month_name[int(month)]