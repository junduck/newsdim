from newsdim.retrieval import Corpus, tokenize


def test_tokenize_company_names():
    tokens = tokenize("平安银行宣布降准50个基点")
    assert "平安银行" in tokens
    assert "降准" in tokens


def test_tokenize_filters_short():
    tokens = tokenize("央行宣布降准")
    for t in tokens:
        assert len(t) >= 2


def test_corpus_add_and_len():
    c = Corpus()
    assert len(c) == 0
    c.add("央行宣布降准50个基点")
    assert len(c) == 1
    c.add_batch(["煤炭板块盘初走强", "财政部公布普惠金融示范区名单"])
    assert len(c) == 3


def test_corpus_get():
    c = Corpus()
    c.add("央行宣布降准50个基点，释放长期资金")
    c.add("煤炭板块盘初走强，大有能源涨停")
    c.add("财政部公布2025年普惠金融发展示范区名单")

    hits = c.get(["降准"], top_k=3)
    assert len(hits) >= 1
    assert hits[0][0] == 0
    assert hits[0][1] > 0


def test_corpus_get_no_match():
    c = Corpus()
    c.add("煤炭板块盘初走强")
    hits = c.get(["降准"], top_k=3)
    assert hits == []


def test_corpus_get_empty():
    c = Corpus()
    hits = c.get(["降准"], top_k=3)
    assert hits == []


def test_corpus_get_empty_keywords():
    c = Corpus()
    c.add("央行宣布降准")
    hits = c.get([], top_k=3)
    assert hits == []


def test_corpus_score_all():
    c = Corpus()
    c.add("央行宣布降准50个基点，释放长期资金")
    c.add("煤炭板块盘初走强，大有能源涨停")
    c.add("财政部公布2025年普惠金融发展示范区名单")

    scores = c.score_all(["降准"])
    assert len(scores) == 3
    assert scores[0] > 0
    assert scores[1] == 0
    assert scores[2] == 0


def test_corpus_roundtrip():
    c = Corpus()
    c.add("央行宣布降准50个基点")
    c.add("煤炭板块盘初走强，大有能源涨停")
    c.add("财政部公布普惠金融示范区名单")

    records = c.to_records()
    assert len(records) == 3
    assert records[0]["text"] == "央行宣布降准50个基点"
    assert "降准" in records[0]["tokens"]

    c2 = Corpus.from_records(records)
    assert len(c2) == 3

    hits1 = c.get(["降准"], top_k=3)
    hits2 = c2.get(["降准"], top_k=3)
    assert hits1 == hits2


def test_corpus_merge_via_records():
    c1 = Corpus()
    c1.add("央行宣布降准50个基点")

    c2 = Corpus()
    c2.add("煤炭板块盘初走强")

    merged = Corpus.from_records(c1.to_records() + c2.to_records())
    assert len(merged) == 2

    hits = merged.get(["降准"], top_k=2)
    assert len(hits) == 1
    assert hits[0][0] == 0


def test_corpus_score_all_consistent_with_get():
    c = Corpus()
    c.add("央行宣布降准50个基点")
    c.add("煤炭板块盘初走强")
    c.add("降准降息预期升温")

    scores = c.score_all(["降准"])
    hits = c.get(["降准"], top_k=3)

    for idx, score in hits:
        assert abs(scores[idx] - score) < 1e-6


def test_corpus_add_invalidates_index():
    c = Corpus()
    c.add("央行宣布降准50个基点")
    hits1 = c.get(["降准"], top_k=1)
    assert len(hits1) == 1

    c.add("再次降准刺激经济")
    hits2 = c.get(["降准"], top_k=2)
    assert len(hits2) == 2


def test_corpus_add_batch_empty():
    c = Corpus()
    c.add_batch([])
    assert len(c) == 0


def test_corpus_score_all_empty():
    c = Corpus()
    scores = c.score_all(["降准"])
    assert scores == []
