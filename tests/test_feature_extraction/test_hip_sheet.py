from neigh_ai.feature_extraction.hip_sheet import get_family_tree


def test_family_tree_complete():
    tree = get_family_tree()

    # Check that all horses are present
    expected_names = ["Thunder Gulch", "Personal Lady", "Hennessy", "Randie's Legend"]
    for name in expected_names:
        assert name in tree

    # Check that parental links are filled correctly
    assert tree["Thunder Gulch"]["sire"] == "Hennessy"
    assert tree["Thunder Gulch"]["dam"] == "Personal Lady"

    # Grandparent propagation
    assert tree["Thunder Gulch"]["paternal_grandsire"] == "Storm Cat"
    assert tree["Thunder Gulch"]["paternal_granddam"] == "N/A"
    assert tree["Thunder Gulch"]["maternal_grandsire"] == "Smart Strike"
    assert tree["Thunder Gulch"]["maternal_granddam"] == "Randie's Legend"

    # Missing entries remain "N/A"
    assert tree["Hennessy"]["maternal_granddam"] == "N/A"
    assert tree["Randie's Legend"]["maternal_grandsire"] == "N/A"
