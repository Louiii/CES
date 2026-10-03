import numpy as np

from nanochat.causal_serialisation_extended import Delete, Insert, Move, Replace, SequenceEdit
from nanochat.edit_dataset import (
    _canonicalize_edit,
    _build_sequence_edit,
    parse_operation_lines,
    sequence_edit_from_json,
    sequence_edit_to_json,
)
from ces.workflows.filter_gec_posttrain_edits import rejection_reason


def test_sequence_edit_json_roundtrip_preserves_edit_types_and_offsets():
    seq = SequenceEdit(
        sequence=np.array(list("abcdefghij")),
        edits=[
            Insert(2, np.array(list("XX"))),
            Delete(3, 5),
            Replace(5, 7, np.array(list("YY"))),
            Move(7, 9, 1),
        ],
    )

    payload = sequence_edit_to_json(seq)
    recovered = sequence_edit_from_json("abcdefghij", payload)

    assert len(recovered.edits) == len(seq.edits)
    for got, want in zip(recovered.edits, seq.edits):
        assert type(got) is type(want)
        if hasattr(got, "content"):
            assert got[:-1] == want[:-1]
            assert np.array_equal(got.content, want.content)
        else:
            assert got == want


def test_parse_operation_lines_supports_region_and_pointer_dsl():
    raw = """
Insert(pointer=Pointer(search_phrase="abc def", pre="abc", post=" def"), text="ZZ")
Replace(region=RegionInclusive(start_search_phrase="abc def", start_exact="abc", end_search_phrase="def ghi", end_exact="def"), text="YY")
""".strip()

    specs, dropped = parse_operation_lines(raw)

    assert dropped == []
    assert specs[0].kwargs["pointer"]["kind"] == "Pointer"
    assert specs[0].kwargs["pointer"]["fields"]["pre"] == "abc"
    assert specs[0].kwargs["pointer"]["fields"]["post"] == " def"
    assert specs[1].kwargs["region"]["kind"] == "RegionInclusive"
    assert specs[1].kwargs["text"] == "YY"


def test_build_sequence_edit_uses_before_and_after_pointer_boundaries():
    source = "alpha beta gamma"
    raw = """
Insert(pointer=Pointer(search_phrase="alpha beta", pre="alpha", post=" beta"), text=" X")
Move(region=RegionInclusive(start_search_phrase="alpha beta", start_exact="alpha", end_search_phrase="alpha beta", end_exact="alpha"), pointer=Pointer(search_phrase="beta gamma", pre="beta gamma", post=""))
""".strip()

    specs, dropped = parse_operation_lines(raw)
    assert dropped == []
    sequence_edit, localise_drops = _build_sequence_edit(source, specs)
    assert localise_drops == []

    insert_edit = next(edit for edit in sequence_edit.edits if isinstance(edit, Insert))
    move_edit = next(edit for edit in sequence_edit.edits if isinstance(edit, Move))

    assert isinstance(insert_edit, Insert)
    assert insert_edit.pos == len("alpha")

    assert isinstance(move_edit, Move)
    assert move_edit.start == 0
    assert move_edit.dest == len(source)


def test_gec_exact_dsl_localizes_a_single_word_without_overlapping_regions():
    source = "She walks to school every morning."
    raw = (
        'ReplaceExact(search_phrase="She walks to school every morning", '
        'exact="walks", text="walk")'
    )
    specs, dropped = parse_operation_lines(raw)
    assert dropped == []
    sequence_edit, localise_drops = _build_sequence_edit(source, specs)
    assert localise_drops == []
    edit = sequence_edit.edits[0]
    # Canonicalisation trims the shared `walk` prefix, leaving deletion of `s`.
    assert isinstance(edit, Delete)
    assert source[edit.start:edit.end] == "s"


def test_gec_filter_rejects_broad_destructive_edits_but_keeps_local_ones():
    common = {
        "num_edits_kept": 1,
        "forward_edit_counts_json": '{"replace": 1, "move": 0}',
    }
    thresholds = {
        "max_edits": 2,
        "max_span_chars": 64,
        "max_distance_chars": 64,
        "max_distance_ratio": 0.15,
    }
    local = common | {
        "source_text": "She walks to school every morning.",
        "corrupted_text": "She walk to school every morning.",
        "edit_span_max": 1,
    }
    destructive = common | {
        "source_text": "A useful clause that should remain intact in the sentence.",
        "corrupted_text": "A useful clause.",
        "edit_span_max": 45,
    }
    assert rejection_reason(local, **thresholds) is None
    assert rejection_reason(destructive, **thresholds) == "distance_ratio_too_large"

    three_edits = local | {"num_edits_kept": 3}
    assert rejection_reason(three_edits, **thresholds) == "edit_count"
    assert rejection_reason(three_edits, **(thresholds | {"max_edits": 4})) is None


def test_region_end_phrase_resolves_after_start_anchor():
    source = (
        "prefix shrinkage. "
        "The resin masks the fiber's scales and keeps them from interlocking, "
        "which is the cause of felting and shrinkage."
    )
    raw = """
Del(region=RegionInclusive(start_search_phrase="The resin masks the fiber's scales and keeps them from interlocking, which is the cause of felting and shrinkage.", start_exact="The", end_search_phrase="shrinkage.", end_exact="shrinkage."))
""".strip()

    specs, dropped = parse_operation_lines(raw)
    assert dropped == []
    sequence_edit, localise_drops = _build_sequence_edit(source, specs)
    assert localise_drops == []
    delete_edit = sequence_edit.edits[0]
    assert isinstance(delete_edit, Delete)
    assert delete_edit.start == source.index("The resin masks")
    assert delete_edit.end == len(source)


def test_canonicalize_replace_trims_shared_prefix_and_suffix():
    source = "abcXYZdef"
    edit = Replace(0, len(source), np.array(list("abc123def")))
    canonical = _canonicalize_edit(source, edit)
    assert isinstance(canonical, Replace)
    assert canonical.start == 3
    assert canonical.end == 6
    assert "".join(np.array(canonical.content).tolist()) == "123"


def test_canonicalize_replace_to_delete_insert_or_identity():
    source = "abcdef"

    delete_like = _canonicalize_edit(source, Replace(2, 4, np.array(list(""))))
    assert isinstance(delete_like, Delete)
    assert delete_like == Delete(2, 4)

    insert_like = _canonicalize_edit(source, Replace(3, 3, np.array(list("ZZ"))))
    assert isinstance(insert_like, Insert)
    assert insert_like.pos == 3
    assert "".join(np.array(insert_like.content).tolist()) == "ZZ"

    identity = _canonicalize_edit(source, Replace(1, 3, np.array(list("bc"))))
    assert identity is None


def test_canonicalize_empty_insert_drops_edit():
    source = "abcdef"
    empty_insert = _canonicalize_edit(source, Insert(2, np.array(list(""))))
    assert empty_insert is None
