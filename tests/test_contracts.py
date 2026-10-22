import math
import pytest
from pydantic import ValidationError
from voxstage.contracts import Scene
from voxstage.text import normalize,compare

@pytest.mark.parametrize('change',[{'start':float('nan')},{'end':float('inf')},{'start':1.4,'end':1.5},
                                  {'end':50},{'character':'missing'},{'id':'../secret'},{'command':'rm'}])
def test_invalid_cue(scene,change):
    value=scene.model_dump(mode='json');value['cues'][0].update(change)
    with pytest.raises(ValidationError):Scene.model_validate(value)

@pytest.mark.parametrize('case',['duplicate_cue','duplicate_cast','overlap','unsorted','speaker','policy_nan'])
def test_graph_contracts(scene,case):
    value=scene.model_dump(mode='json')
    if case=='duplicate_cue':value['cues'][1]['id']='one'
    if case=='duplicate_cast':value['cast']*=2
    if case=='overlap':value['cues'][1]['start']=1
    if case=='unsorted':value['cues'].reverse()
    if case=='speaker':value['cast'][0]['speaker']='arbitrary_person'
    if case=='policy_nan':value['policy']['max_cer']=math.nan
    with pytest.raises(ValidationError):Scene.model_validate(value)

def test_normalization_does_not_erase_negation_or_numbers():
    assert normalize('不要打开，３号门！')=='不要打开3号门'
    assert compare('不要打开三号门','打开三号门')['cer']>0
    assert compare('gate 12','gate 21')['cer']>0
    assert compare('Hello, WORLD!','hello world')['cer']==0

def test_cer_insertions_can_exceed_one():
    assert compare('a','abcdefgh')['cer']==7
    with pytest.raises(ValueError):compare('!!!','hello')

def test_explicit_speech_form_preserves_display(scene):
    value=scene.model_dump(mode='json');value['cues'][0]['speech_text']='不要打开三号门。'
    revised=Scene.model_validate(value)
    assert revised.cues[0].text==scene.cues[0].text
    assert revised.cues[0].spoken=='不要打开三号门。'
