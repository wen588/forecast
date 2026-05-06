# -*- coding: utf-8 -*-

import logging
import os
from datetime import datetime


class Logger:
    """
    工程级日志系统（每次运行生成唯一log文件）
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
        self.log_name = log_name
        self.level = level
        self.fmt = fmt
        self.console = console

        self.logger = logging.getLogger(self._get_unique_name())
        self.logger.setLevel(self.level_relations[level])

        # ⭐关键：避免重复 handler
        if not self.logger.handlers:
            self._setup_logger()

    def _get_unique_name(self):
        """
        防止 logging.getLogger 重复污染
        """
        return f"{self.log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _setup_logger(self):

        # ========= log目录 =========
        log_dir = os.path.join(self.root_path, "log")
        os.makedirs(log_dir, exist_ok=True)

        # ⭐核心：每次运行唯一文件名
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{self.log_name}_{time_str}.log")

        formatter = logging.Formatter(self.fmt)

        # ========= File Handler =========
        file_handler = logging.FileHandler(
            log_file,
            mode="w",   # ⭐关键：每次运行覆盖写（不会追加旧日志）
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # ========= Console Handler =========
        if self.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger