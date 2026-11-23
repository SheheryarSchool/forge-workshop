"""
Tests for the alignment check and scoring in notebooks/sheheryar/eval.py.

    python tests/test_alignment.py        (no extra packages needed)
    pytest tests/                          (also works)

Most tests run against the CSVs in outputs/predictions/ (the real files, once committed).
In a fresh clone with no CSVs yet, they generate dummy predictions in a temp folder instead,
so nothing is ever written into outputs/predictions/.
test_official_split_round_trip is self-contained: it builds a tiny meld_clean.csv, saves
predictions the way save_predictions() does, and checks the path that runs on the day.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "sheheryar"))
import eval as ev  # noqa: E402


def _pick_pred_dir() -> Path:
    committed = ROOT / "outputs" / "predictions"
    if any(committed.glob("*.csv")):
        return committed
    tmp = Path(tempfile.mkdtemp(prefix="forge-test-dummies-"))
    subprocess.run([sys.executable, str(ROOT / "notebooks" / "sheheryar" / "make_dummy_predictions.py"),
                    "--pred-dir", str(tmp / "predictions"), "--fixtures-dir", str(tmp / "fixtures")],
                   check=True, capture_output=True)
    print(f"(no CSVs in outputs/predictions/ yet: testing on dummy data generated in {tmp})")
    return tmp / "predictions"


PRED_DIR = _pick_pred_dir()
FIXTURE = ROOT / "tests" / "fixtures" / "misaligned_bert_finetuned_predictions.csv"


def fresh():
    return ev.load_predictions(PRED_DIR)


def official():
    return ev.load_official_test(ROOT)


def expect_failure(frames, *must_contain: str, official_test=None, dummy_files=frozenset()) -> str:
    try:
        ev.check_alignment(frames, official_test if official_test is not None else official(), dummy_files)
    except ev.AlignmentError as err:
        msg = str(err)
        for text in must_contain:
            assert text in msg, f"expected {text!r} in the error, got:\n{msg}"
        return msg
    raise AssertionError("alignment check PASSED but should have FAILED")


def swap_rows(df: pd.DataFrame, start: int = 1233) -> int:
    """Swap two neighbouring rows whose utterances differ; return the first index."""
    i = next(j for j in range(start, len(df) - 1) if df["utterance"].iloc[j] != df["utterance"].iloc[j + 1])
    df.iloc[[i, i + 1]] = df.iloc[[i + 1, i]].to_numpy()
    return i


# ------------------------------------------------------------------ files in outputs/predictions
def test_good_files_pass():
    frames = fresh()
    result = ev.check_alignment(frames, official(), ev.find_dummy_files(PRED_DIR))
    assert len(result.models) + len(result.set_aside) == len(frames)
    assert not result.warnings, result.warnings


def test_misaligned_fixture_fails():
    frames = fresh()
    frames["bert_finetuned_predictions.csv"] = ev.read_predictions_csv(FIXTURE)
    # Against dummies it fails at row 1233; against the real data it fails earlier. Either way: loud, named.
    expect_failure(frames, "ALIGNMENT CHECK FAILED", "bert_finetuned_predictions.csv")


def test_two_swapped_rows_fail_at_the_right_row():
    frames = fresh()
    name = next(iter(frames))
    i = swap_rows(frames[name])
    expect_failure(frames, name, f"First mismatched row: {i}")


def test_capped_100_row_file_fails():
    frames = fresh()
    name = next(iter(frames))
    frames[name] = frames[name].head(100)
    expect_failure(frames, name, "Has 100 rows", "capped")


def test_uppercase_label_fails():
    frames = fresh()
    name = next(iter(frames))
    frames[name].loc[10, "predicted_label"] = "Anger"
    expect_failure(frames, name, "row 10", "lowercase")


def test_wrong_column_order_fails():
    frames = fresh()
    name = next(iter(frames))
    frames[name] = frames[name][["true_label", "utterance", "predicted_label", "model_name"]]
    expect_failure(frames, name, "Columns are")


def test_unknown_model_name_fails():
    frames = fresh()
    name = next(iter(frames))
    frames[name]["model_name"] = "bert"
    expect_failure(frames, name, "not one of the fixed names")


def test_duplicate_model_fails():
    frames = fresh()
    name, df = next(iter(frames.items()))
    frames["copy_of_" + name] = df.copy()
    expect_failure(frames, "already used by")


def test_missing_model_still_scores():
    frames = fresh()
    frames.pop(next(iter(frames)))
    result = ev.check_alignment(frames, official())
    assert len(ev.evaluate_all(result.frames)) == len(frames)


def test_shuffled_predictions_are_flagged():
    frames = fresh()
    name = next(iter(frames))
    df = frames[name]
    df["predicted_label"] = df["predicted_label"].sample(frac=1, random_state=0).to_numpy()
    result = ev.check_alignment(frames, official())  # rows still line up, so it passes...
    assert any(name in w and "unrelated" in w for w in result.warnings), result.warnings  # ...but warns


# ------------------------------------------------------------------ real files mixed with dummies
def _as_real(frames: dict, names: list[str]) -> None:
    """Turn some dummy frames into stand-ins for real files: different (shared) utterances."""
    for n in names:
        frames[n]["utterance"] = [f"real line {i}" for i in range(len(frames[n]))]


def test_leftover_dummies_never_outvote_real_files():
    frames = fresh()
    names = sorted(frames)
    real, dummies = names[:2], set(names[2:])
    _as_real(frames, real)
    result = ev.check_alignment(frames, None, dummies)
    assert set(result.models) == set(real), result.models
    assert set(result.set_aside) == dummies, result.set_aside


def test_misaligned_real_file_still_fails_next_to_dummies():
    frames = fresh()
    names = sorted(frames)
    real, dummies = names[:3], set(names[3:])
    _as_real(frames, real)
    swap_rows(frames[real[0]])
    expect_failure(frames, real[0], official_test=None, dummy_files=dummies)


def test_no_majority_is_reported_not_guessed():
    frames = fresh()
    a, b = sorted(frames)[:2]
    two = {a: frames[a], b: frames[b]}
    swap_rows(two[b])
    expect_failure(two, "no version has a majority", official_test=None)


# ------------------------------------------------------------------ the path that runs on the day
TRICKY = ["Hello, world", 'She said "hi"', "line one\nline two", "  leading spaces", "trailing  ",
          "1", "café ’quote’", " nbsp", "Okay.", "Okay."]


def test_official_split_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "data" / "processed").mkdir(parents=True)
        (tmp / "preds").mkdir()
        rng = np.random.default_rng(0)
        n = 60
        utterances = [TRICKY[i % len(TRICKY)] + f" #{i}" if i % 3 else TRICKY[i % len(TRICKY)] for i in range(n)]
        meld = pd.DataFrame({
            "utterance": utterances,
            "label": rng.choice(ev.LABELS, n),
            "dialogue_id": [i // 6 for i in range(n)],
            "utterance_id": [i % 6 for i in range(n)],
            "speaker": "A",
            "split": ["train"] * 24 + ["dev"] * 12 + ["Test"] * 24,
        })
        meld.to_csv(tmp / "data" / "processed" / "meld_clean.csv", index=False)

        # Exactly how the base notebook reads, splits and saves (Step 4, Step 6, save_predictions)
        base = pd.read_csv(tmp / "data" / "processed" / "meld_clean.csv")
        test_df = base[base["split"].str.lower() == "test"].reset_index(drop=True)
        X_test, y_test = test_df["utterance"].tolist(), test_df["label"].tolist()
        for model in ("bert_finetuned", "zero_shot"):
            pd.DataFrame({"utterance": X_test, "true_label": y_test,
                          "predicted_label": rng.choice(ev.LABELS, len(X_test)), "model_name": model}
                         ).to_csv(tmp / "preds" / f"{model}_predictions.csv", index=False)

        frames = ev.load_predictions(tmp / "preds")
        result = ev.check_alignment(frames, ev.load_official_test(tmp))
        assert result.used_official_test and result.n_rows == 24 and len(result.models) == 2

        i = swap_rows(frames["zero_shot_predictions.csv"], start=3)
        expect_failure(frames, "zero_shot_predictions.csv", f"First mismatched row: {i}",
                       official_test=ev.load_official_test(tmp))


def test_unreadable_file_is_named():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bilstm_predictions.csv"
        bad.write_bytes(b"utterance,true_label,predicted_label,model_name\n\x92oops,anger,anger,bilstm\n")
        try:
            ev.load_predictions(tmp)
        except ev.AlignmentError as err:
            assert "bilstm_predictions.csv" in str(err)
            return
        raise AssertionError("an unreadable file should raise AlignmentError naming it")


# ------------------------------------------------------------------ metrics
def test_bootstrap_macro_f1_matches_sklearn():
    frames = fresh()
    r = ev.evaluate_all(frames)[0]
    lut = {label: i for i, label in enumerate(ev.LABELS)}
    t = np.array([lut[x] for x in r.y_true])
    p = np.array([lut[x] for x in r.y_pred])
    same_rows = np.arange(len(t))[None, :]
    assert abs(ev._macro_f1_resampled(t, p, same_rows)[0] - r.macro_f1) < 1e-12


def test_majority_baseline_is_neutral_and_scores_low_macro_f1():
    y_true = next(iter(fresh().values()))["true_label"].tolist()
    b = ev.majority_baseline(y_true)
    assert b.model_name == "always_neutral"
    assert b.macro_f1 < 0.15


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # report every failure, don't stop at the first
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
