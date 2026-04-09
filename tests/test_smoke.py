"""Placeholder to keep pytest happy until real tests are added."""


def test_import() -> None:
    """Verify the package is importable."""
    import torchquant

    assert torchquant.__name__ == "torchquant"
