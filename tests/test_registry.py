"""
Tests for the component registry.
"""

import pytest

from flashrag.registry import EMBEDDINGS, GENERATORS, PIPELINES, RERANKERS, RETRIEVERS, _Registry


class TestRegistry:
    def test_create_registry(self):
        reg = _Registry("TEST")
        assert reg.name == "TEST"
        assert reg.list() == []

    def test_register_and_get(self):
        reg = _Registry("TEST")

        @reg.register("my_component")
        class MyComponent:
            pass

        assert "my_component" in reg
        assert reg.get("my_component") is MyComponent

    def test_register_default_name(self):
        reg = _Registry("TEST")

        @reg.register()
        class AnotherComponent:
            pass

        assert "AnotherComponent" in reg

    def test_get_missing_raises(self):
        reg = _Registry("TEST")
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_duplicate_registration_raises(self):
        reg = _Registry("TEST")

        @reg.register("dup")
        class First:
            pass

        with pytest.raises(ValueError, match="already contains"):

            @reg.register("dup")
            class Second:
                pass

    def test_list_keys(self):
        reg = _Registry("TEST")

        @reg.register("a")
        class A:
            pass

        @reg.register("b")
        class B:
            pass

        keys = reg.list()
        assert "a" in keys
        assert "b" in keys

    def test_contains(self):
        reg = _Registry("TEST")

        @reg.register("exists")
        class Exists:
            pass

        assert "exists" in reg
        assert "missing" not in reg

    def test_repr(self):
        reg = _Registry("MY_REG")
        assert "MY_REG" in repr(reg)

    def test_global_registries_exist(self):
        assert isinstance(EMBEDDINGS, _Registry)
        assert isinstance(RETRIEVERS, _Registry)
        assert isinstance(GENERATORS, _Registry)
        assert isinstance(PIPELINES, _Registry)
        assert isinstance(RERANKERS, _Registry)

    def test_auto_register(self):
        from flashrag.registry import auto_register

        auto_register()
        assert "faiss" in RETRIEVERS
        assert "bm25" in RETRIEVERS
        assert "sentence_transformer" in EMBEDDINGS
        assert "basic_rag" in PIPELINES
