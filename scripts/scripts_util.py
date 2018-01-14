import logging
import re

import scandir


def initLogging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def scan_dir(dir_path, pattern=None, dir_level=-1):
    def inner_scan_project(in_path, level=0):
        for entry in scandir.scandir(in_path):
            if dir_level == level:
                if pattern is None:
                    yield entry.path
                else:
                    if pattern(entry.name):
                        yield entry.path
                continue
            if entry.is_dir():
                yield from inner_scan_project(entry.path, level=level+1)
            elif entry.is_file():
                if pattern is None:
                    yield entry.path
                else:
                    if pattern(entry.name):
                        yield entry.path
    yield from inner_scan_project(dir_path)


def remove_comments(code):
    pattern = r"(\".*?(?<!\\)\"|\'.*?(?<!\\)\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return "" # so we will return empty to remove the comment
        else: # otherwise, we will return the 1st group
            return match.group(1) # captured quoted-string
    return regex.sub(_replacer, code)


def remove_blank_line(code):
    code = "\n".join([line for line in code.split('\n') if line.strip() != ''])
    return code


def remove_r_char(code):
    code = code.replace('\r', '')
    return code


def remove_blank(code):
    pattern = re.compile('''('.*?'|".*?"|[^ \t\r\f\v"']+)''')
    mat = re.findall(pattern, code)
    processed_code = ' '.join(mat)
    return processed_code