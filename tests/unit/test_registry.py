import pytest
from pydantic import BaseModel

from mtmct import registry


@registry.register("solver", "_test_dummy")
class _Dummy:
    """Dummy solver for tests."""

    LICENCE = "MIT"

    class Params(BaseModel):
        threshold: float = 0.5

    def __init__(self, params: "_Dummy.Params"):
        self.params = params


def test_registry_create_and_catalog_schema():
    obj = registry.create("solver", "_test_dummy", {"threshold": 0.7})
    assert obj.params.threshold == 0.7
    info = {i.name: i for i in registry.catalog()["solver"]}["_test_dummy"]
    assert info.params_schema["properties"]["threshold"]["default"] == 0.5
    assert info.licence == "MIT" and info.description == "Dummy solver for tests."


def test_registry_rejects_missing_params_and_unknown():
    with pytest.raises(TypeError):

        @registry.register("solver", "_bad")
        class _Bad:
            LICENCE = "MIT"

    with pytest.raises(KeyError):
        registry.get("solver", "_nope")
    with pytest.raises(ValueError):
        registry.register("nope", "x")
