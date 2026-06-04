from wc26 import news


def test_extract_json_array_strips_fences():
    assert news._extract_json_array('```json\n[{"player": "X"}]\n```') == [{"player": "X"}]


def test_extract_json_array_finds_embedded():
    assert news._extract_json_array("Here you go: [1, 2, 3]  done.") == [1, 2, 3]


def test_extract_json_array_empty_on_garbage():
    assert news._extract_json_array("no json at all") == []
    assert news._extract_json_array("[broken") == []


def test_extract_json_array_handles_stray_brackets():
    s = 'I checked [sources] and found: [{"player": "X", "status": "out"}]'
    assert news._extract_json_array(s) == [{"player": "X", "status": "out"}]
