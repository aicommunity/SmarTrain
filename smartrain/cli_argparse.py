"""Общий ArgumentParser: в help показываются значения по умолчанию."""

from __future__ import annotations

import argparse


class CliArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*args, **kwargs)
