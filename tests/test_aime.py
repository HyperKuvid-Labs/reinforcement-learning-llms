from llmrl.aime import compute_binary_reward, extract_final_answer, normalize_answer


def test_extract_boxed_answer():
    assert extract_final_answer("work\n\\boxed{42}") == "42"


def test_extract_last_number_fallback():
    assert extract_final_answer("Therefore the answer is 384.") == "384"


def test_normalize_answer_numeric():
    assert normalize_answer("0042") == "42"
    assert normalize_answer("1,200") == "1200"


def test_binary_reward_uses_normalized_final_answer():
    assert compute_binary_reward("steps...\n\\boxed{070}", "70") == 1
