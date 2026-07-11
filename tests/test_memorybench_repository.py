from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_breaking_namespace_and_entrypoint_cleanup():
    assert not (ROOT / "src/amem").exists()
    assert not list(ROOT.glob("*.py"))
    assert not list((ROOT / "scripts").glob("*.py"))
    offenders = []
    for base in (ROOT / "src", ROOT / "tests"):
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if ("from " + "amem") in text or ("import " + "amem") in text:
                offenders.append(path)
    assert offenders == []


def test_shell_wrappers_only_call_memorybench_cli():
    wrappers = list((ROOT / "scripts").glob("*.sh"))
    assert wrappers
    assert all("python -m memorybench" in path.read_text(encoding="utf-8") for path in wrappers)
