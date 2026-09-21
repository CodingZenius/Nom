#!/usr/bin/env python3
"""Entry point:  python run.py [onboard|boot|doctor|chat|run|resume|shot|sync]"""
import sys

from nomad.cli import main

if __name__ == "__main__":
    sys.exit(main())
