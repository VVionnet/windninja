/*
 * Copyright (c) 2001-2006, NLnet Labs. All rights reserved.
 *
 * This software is open source.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * Neither the name of the NLNET LABS nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef USE_INTERNAL_TIME_GM
#include <time.h>

/* Number of days per month (except for February in leap years). */
static const int monoff[] = {
        0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334
};

static int
is_leap_year(int year)
{
        return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

static int
leap_days(int y1, int y2)
{
        --y1;
        --y2;
        return (y2/4 - y1/4) - (y2/100 - y1/100) + (y2/400 - y1/400);
}

/*
 * Code adapted from Python 2.4.1 sources (Lib/calendar.py).
 */
time_t
timegm(const struct tm *tm)
{
        int year;
        time_t days;
        time_t hours;
        time_t minutes;
        time_t seconds;

        year = 1900 + tm->tm_year;
        days = 365 * (year - 1970) + leap_days(1970, year);
        days += monoff[tm->tm_mon];

        if (tm->tm_mon > 1 && is_leap_year(year))
                ++days;
        days += tm->tm_mday - 1;

        hours = days * 24 + tm->tm_hour;
        minutes = hours * 60 + tm->tm_min;
        seconds = minutes * 60 + tm->tm_sec;

        return seconds;
}
/*
** Straight from man timegm:
** NOTES
**        The timelocal() function is equivalent to the POSIX standard function mktime(3).  There is no reason to ever use it.
** 
**        For a portable version of timegm(), set the TZ environment variable to UTC, call mktime(3) and restore the value of TZ.  Something like
** 
**            #include <time.h>
**            #include <stdlib.h>
** 
**            time_t
**            my_timegm(struct tm *tm)
**            {
**                time_t ret;
**                char *tz;
** 
**                tz = getenv("TZ");
**                setenv("TZ", "", 1);
**                tzset();
**                ret = mktime(tm);
**                if (tz)
**                    setenv("TZ", tz, 1);
**                else
**                    unsetenv("TZ");
**                tzset();
**                return ret;
**           }
*/

#endif /* USE_INTERNAL_TIME_GM */

