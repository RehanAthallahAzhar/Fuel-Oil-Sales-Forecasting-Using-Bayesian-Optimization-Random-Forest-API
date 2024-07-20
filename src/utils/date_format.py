from datetime import datetime

import calendar

# CONST FORMATDATETODAY= "%d"

def FormatTimestampToDay(timestamp):
    return datetime.fromtimestamp(timestamp).day

def ConvertMonthtoLatin(month):
    return calendar.month_name[int(month)] 

