#!/usr/bin/env python

"""Tests for `taufactor` package."""

import pytest

from click.testing import CliRunner

import taufactor


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_solver(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    S = taufactor.Solver()

