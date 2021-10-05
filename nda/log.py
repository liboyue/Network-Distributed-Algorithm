# Cloned from https://github.com/benley/python-glog
"""A simple Google-style logging wrapper."""

import os
import sys
import time
import traceback
import logging
import colorlog


def format_message(record):
    try:
        record_message = '%s' % (record.msg % record.args)
    except TypeError:
        record_message = record.msg
    return record_message


class MyLogFormatter(colorlog.ColoredFormatter):
    LEVEL_MAP = {
        logging.FATAL: 'F',  # FATAL is alias of CRITICAL
        logging.ERROR: 'E',
        logging.WARN: 'W',
        logging.INFO: 'I',
        logging.DEBUG: 'D'
    }

    def __init__(self):
        colorlog.ColoredFormatter.__init__(self, '%(log_color)s%(levelname)s %(message)s%(reset)s')

    def format(self, record):
        date = time.localtime(record.created)
        date_usec = (record.created - int(record.created)) * 1e4
        record_message = '%02d:%02d:%02d.%04d %s %s:%d] %s' % (
            date.tm_hour, date.tm_min,
            date.tm_sec, date_usec,
            record.process if record.process is not None else '?????',
            record.filename,
            record.lineno,
            format_message(record))
        record.getMessage = lambda: record_message
        return colorlog.ColoredFormatter.format(self, record)


def set_level(new_level):
    logger.setLevel(new_level)
    logger.debug('Log level set to %s', new_level)


debug = logging.debug
info = logging.info
warning = logging.warning
warn = logging.warning
error = logging.error
exception = logging.exception
fatal = logging.fatal
log = logging.log

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
WARN = logging.WARN
ERROR = logging.ERROR
FATAL = logging.FATAL

handler = logging.StreamHandler()
handler.setFormatter(MyLogFormatter())

glog = logger = logging.getLogger()
logger.addHandler(handler)
set_level('INFO')


def _critical(self, message, *args, **kws):
    self._log(50, message, args, **kws)
    sys.exit(-1)


logging.Logger.critical = _critical

# Define functions emulating C++ glog check-macros
# https://htmlpreview.github.io/?https://github.com/google/glog/master/doc/glog.html#check


def format_stacktrace(stack):
    """Print a stack trace that is easier to read.

    * Reduce paths to basename component
    * Truncates the part of the stack after the check failure
    """
    lines = []
    for _, f in enumerate(stack):
        fname = os.path.basename(f[0])
        line = "\t%s:%d\t%s" % (fname + "::" + f[2], f[1], f[3])
        lines.append(line)
    return lines


class FailedCheckException(AssertionError):
    """Exception with message indicating check-failure location and values."""


def check_failed(message):
    stack = traceback.extract_stack()
    stack = stack[0:-2]
    stacktrace_lines = format_stacktrace(stack)
    filename, line_num, _, _ = stack[-1]

    try:
        raise FailedCheckException(message)
    except FailedCheckException:
        log_record = logger.makeRecord('CRITICAL', 50, filename, line_num,
                                       message, None, None)
        handler.handle(log_record)

        log_record = logger.makeRecord('DEBUG', 10, filename, line_num,
                                       'Check failed here:', None, None)
        handler.handle(log_record)
        for line in stacktrace_lines:
            log_record = logger.makeRecord('DEBUG', 10, filename, line_num,
                                           line, None, None)
            handler.handle(log_record)
        raise


def check(condition, message=None):
    """Raise exception with message if condition is False."""
    if not condition:
        if message is None:
            message = "Check failed."
        check_failed(message)


def check_eq(obj1, obj2, message=None):
    """Raise exception with message if obj1 != obj2."""
    if obj1 != obj2:
        if message is None:
            message = "Check failed: %s != %s" % (str(obj1), str(obj2))
        check_failed(message)


def check_ne(obj1, obj2, message=None):
    """Raise exception with message if obj1 == obj2."""
    if obj1 == obj2:
        if message is None:
            message = "Check failed: %s == %s" % (str(obj1), str(obj2))
        check_failed(message)


def check_le(obj1, obj2, message=None):
    """Raise exception with message if not (obj1 <= obj2)."""
    if obj1 > obj2:
        if message is None:
            message = "Check failed: %s > %s" % (str(obj1), str(obj2))
        check_failed(message)


def check_ge(obj1, obj2, message=None):
    """Raise exception with message unless (obj1 >= obj2)."""
    if obj1 < obj2:
        if message is None:
            message = "Check failed: %s < %s" % (str(obj1), str(obj2))
        check_failed(message)


def check_lt(obj1, obj2, message=None):
    """Raise exception with message unless (obj1 < obj2)."""
    if obj1 >= obj2:
        if message is None:
            message = "Check failed: %s >= %s" % (str(obj1), str(obj2))
        check_failed(message)


def check_gt(obj1, obj2, message=None):
    """Raise exception with message unless (obj1 > obj2)."""
    if obj1 <= obj2:
        if message is None:
            message = "Check failed: %s <= %s" % (str(obj1), str(obj2))
        check_failed(message)


def check_notnone(obj, message=None):
    """Raise exception with message if obj is None."""
    if obj is None:
        if message is None:
            message = "Check failed: Object is None."
        check_failed(message)
