import os
from amber import architect
from amber.utils import testing_utils
import numpy as np

def test_mock_reward():
    hist_file_list = [os.path.join(os.path.dirname(__file__),"../integration/", "mock_black_box/tmp_%i/train_history.csv.gz" % i)
                               for i in range(1, 21)]
    reward_fn = architect.reward.MockReward(
        train_history_list=hist_file_list,
        metric=['loss', 'knowledge'],
        stringify_states=True,
        metric_name_dict={'loss':0, 'knowledge':1, 'acc':2}
    )
    model_space = testing_utils.get_bionas_model_space()
    arc = [l[0] for l in model_space]
    reward = reward_fn(arc)
    assert len(reward) == 3


def test_f1_reward():
    reward_fn = architect.reward.F1Reward()
    # test integer target, integer pred
    reward = reward_fn.call_scorer(y=[0,1,2], pred=[1,0,2])
    assert np.isscalar(reward[0][0]) and len(reward)==1, f"got {reward} for test 1"
    # test integer target, matrix pred
    reward = reward_fn.call_scorer(y=[0,1,2], pred=np.random.randn(3,3))
    assert np.isscalar(reward[0][0]) and len(reward)==1, f"got {reward} for test 2"
    # test matrix target, matrix pred
    reward = reward_fn.call_scorer(y=np.eye(3)[[0,1,2]], pred=np.random.randn(3,3))
    assert np.isscalar(reward[0][0]) and len(reward)==1, f"got {reward} for test 3"


if __name__ == '__main__':
    test_f1_reward()
