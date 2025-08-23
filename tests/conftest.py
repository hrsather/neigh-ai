from unittest.mock import Mock

import dash

dash.register_page = Mock()

from pytest import fixture  # noqa: E402

from neigh_ai.dashboard.pages.scorecard import Scorecard  # noqa: E402


@fixture
def scorecard():
    return Scorecard()
