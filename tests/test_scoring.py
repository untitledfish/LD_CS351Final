from src.scoring.scoring_engine import ScoringEngine
import json, tempfile

def test_eval():
    cfg = {
        "classes": {"robot":0, "blue_shooting":1},
        "rules": [{"name":"shoot","class_trigger":"blue_shooting","score":3,"frames_confirm":1}],
        "zones": {}
    }
    tf = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
    import json
    json.dump(cfg, tf)
    tf.close()
    se = ScoringEngine(tf.name)
    evs = se.evaluate(1, [{'class_name':'red_shooting','center':(10,10)}], 1)
    assert len(evs) == 1
    assert evs[0]['points'] == 3
