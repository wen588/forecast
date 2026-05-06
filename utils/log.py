# -*- coding: utf-8 -*-

import logging
import os
from datetime import datetime

class Logger:
    """
    工程级日志系统（文件名：model_time.log）
    """

    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self,
                 root_path: str,
                 log_name: str = "train",
                 level: str = "info",
                 fmt: str = "%(asctime)s | %(levelname)s | %(message)s",
                 console: bool = True):

        self.root_path = root_path
        self.log_name = log_name.split("_")[0]   # ⭐只保留主名（lstm / bpnn / rnn）
        self.level = level
        self.fmt = fmt
        self.console = console

        self.logger = logging.getLogger(self._get_unique_name())
        self.logger.setLevel(self.level_relations[level])

        if not self.logger.handlers:
            self._setup_logger()

    def _get_unique_name(self):
        return f"{self.log_name}_{datetime.now().strftime('%H%M%S')}"

    def _setup_logger(self):

        log_dir = os.path.join(self.root_path, "log")
        os.makedirs(log_dir, exist_ok=True)

        # ⭐只保留：lstm_153012.log
        time_str = datetime.now().strftime("%H%M%S")
        log_file = os.path.join(log_dir, f"{self.log_name}_{time_str}.log")

        formatter = logging.Formatter(self.fmt)

        file_handler = logging.FileHandler(
            log_file,
            mode="w",
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if self.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
