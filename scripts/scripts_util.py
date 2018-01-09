import logging
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