from newsdim.ann_scorer.rules import score_announcement


def test_asset_acquisition():
    r = score_announcement("资产收购", "拟向张富发行股份及支付现金购买其持有的上海佳投100%股权")
    assert r == {"mom": 0, "stab": -1, "horz": 1, "eng": 2, "hype": 0, "sent": 1, "sec": -3, "pol": 0}


def test_share_pledge():
    r = score_announcement("股份质押", "公司股东将持有的本公司股份500万股质押给证券公司")
    assert r == {"mom": 0, "stab": -1, "horz": 0, "eng": 1, "hype": 0, "sent": -1, "sec": -3, "pol": 0}


def test_guarantee_yi():
    r = score_announcement("对外担保", "公司为其提供担保金额人民币2.4亿元")
    assert r == {"mom": 0, "stab": -1, "horz": 0, "eng": 2, "hype": 0, "sent": 0, "sec": -3, "pol": 0}


def test_guarantee_wan():
    r = score_announcement("对外担保", "子公司申请流动资金贷款额度不超过人民币1000万元")
    assert r == {"mom": 0, "stab": -1, "horz": 0, "eng": 1, "hype": 0, "sent": 0, "sec": -3, "pol": 0}


def test_restructure_disposal():
    r = score_announcement("资产重组", "拟通过公开挂牌方式处置控股子公司部分固定资产")
    assert r["eng"] == 2
    assert r["stab"] == -2
    assert r["sent"] == -2
    assert r["sec"] == -3


def test_restructure_capital_increase():
    r = score_announcement("资产重组", "公司拟使用募集资金人民币2亿元对全资子公司增资")
    assert r["eng"] == 1
    assert r["horz"] == 1
    assert r["stab"] == 0
    assert r["sec"] == -3


def test_restructure_new_company():
    r = score_announcement("资产重组", "全资子公司与XX公司共同出资设立合资公司")
    assert r["eng"] == 1
    assert r["horz"] == 1


def test_restructure_acquisition():
    r = score_announcement("资产重组", "拟向交易对方购买其持有的XX公司100%股权")
    assert r["eng"] == 2
    assert r["stab"] == -1
    assert r["horz"] == 1


def test_restructure_default():
    r = score_announcement("资产重组", "公司拟调整业务结构")
    assert r == {"mom": 0, "stab": 0, "horz": 0, "eng": 1, "hype": 0, "sent": 0, "sec": -3, "pol": 0}


def test_unknown_type():
    r = score_announcement("未知类型", "some content")
    assert r == {"mom": 0, "stab": 0, "horz": 0, "eng": 0, "hype": 0, "sent": 0, "sec": 0, "pol": 0}
