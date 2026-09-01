from orchestrator.core.plan_version import PlanVersion


def test_initial_version_has_no_parent():
    v1 = PlanVersion.initial({"tasks": []})
    assert v1.version == 1
    assert v1.parent_plan_id is None
    assert v1.plan_id


def test_next_version_links_back_to_parent_and_never_overwrites():
    v1 = PlanVersion.initial({"tasks": [{"id": "a"}]})
    v2 = v1.next_version({"tasks": [{"id": "a"}, {"id": "b"}]}, change_reason="added b", patch_ops=[{"op": "add_task"}])

    assert v2.version == 2
    assert v2.parent_plan_id == v1.plan_id
    assert v2.plan_id != v1.plan_id
    assert v2.change_reason == "added b"
    assert v2.patch_ops == [{"op": "add_task"}]
    # v1 itself is untouched - a distinct object, never mutated in place
    assert v1.version == 1
    assert v1.graph_snapshot == {"tasks": [{"id": "a"}]}


def test_round_trips_through_dict():
    v1 = PlanVersion.initial({"tasks": []}, change_reason="initial plan")
    v2 = v1.next_version({"tasks": [{"id": "a"}]}, change_reason="replan")

    restored = PlanVersion.from_dict(v2.to_dict())
    assert restored == v2


def test_version_chain_of_three():
    v1 = PlanVersion.initial({"tasks": []})
    v2 = v1.next_version({"tasks": ["b"]}, change_reason="r1")
    v3 = v2.next_version({"tasks": ["c"]}, change_reason="r2")

    assert [v.version for v in (v1, v2, v3)] == [1, 2, 3]
    assert v3.parent_plan_id == v2.plan_id
    assert v2.parent_plan_id == v1.plan_id
