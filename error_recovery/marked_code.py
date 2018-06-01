from common import util
import config
import os
import typing
import re

temp_code_path = config.temp_code_write_path
line_markers_pattern = re.compile(r'# (\d+) "(.+)"( [1-4])*')


class MarkedCode(object):
    def __init__(self,
                 headers: typing.List[str],
                 header_names: typing.List[str],
                 sources: typing.List[str],
                 source_names: typing.List[str]):
        self.headers = headers
        self.header_names = header_names
        self.sources = sources
        self.source_names = source_names
        self._preprocessed_code = self._preprocess()[0]
        self._line_is_in_system_header = self._mark_line_to_header_files_or_source_files()

    def __str__(self):
        """
        :return: the preprocessed token string
        """
        return self._preprocessed_code

    def _preprocess(self) -> typing.List[str]:
        util.make_dir(temp_code_path)

        def write_code_list(dir_name, codes, code_names):
            code_path = os.path.join(temp_code_path, dir_name)
            util.make_dir(code_path)
            for c, n in zip(codes, code_names):
                util.write_code(c, os.path.join(code_path, n))

        write_code_list('source', self.sources, self.source_names)
        write_code_list('header', self.headers, self.header_names)

        return [util.preprocess(os.path.join(temp_code_path, 'source', n),
                                [os.path.join(temp_code_path, 'header')])
                for n in self.source_names]

    def _mark_line_to_header_files_or_source_files(self):
        def split_lines(codes):
            return [c.split('\n') for c in codes]
        splited_preprocessed_codes = split_lines([self._preprocessed_code])[0]
        is_in_system_header_lines = [False] * len(splited_preprocessed_codes)

        is_in_system_header = False
        for i, line in enumerate(splited_preprocessed_codes):
            m = line_markers_pattern.match(line)
            if m:
                line_number = m.group(1)
                file_name = m.group(2)
                flags = m.group(3)
                if flags:
                    name = os.path.split(file_name)[1]
                    flags = int(flags)
                    if flags == 3:
                        is_in_system_header = False
                    elif flags == 1 or flags == 2:
                        if name in self.header_names or name in self.source_names:
                            is_in_system_header = False
                        else:
                            is_in_system_header = True
            is_in_system_header_lines[i] = is_in_system_header

        # for i, (label, code) in enumerate(zip(is_in_system_header_lines, splited_preprocessed_codes)):
        #     print("{},{}:{}".format(i, label, code))
        util.remove_content_in_dir(temp_code_path)
        return is_in_system_header_lines

    def is_in_system_header(self, line_no) -> bool:
        """
        :param line_no: the line number (start from 1)
        :type line_no: int
        :return: a boolean variable indicating whether the line number in the preprocessed file is originally in the
        system header
        """
        if line_no > len(self._line_is_in_system_header):
            raise ValueError("The line number {} is out of the preprocessed code".format(line_no))
        return self._line_is_in_system_header[line_no - 1]
