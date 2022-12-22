import os
from amber import architect
from amber.utils import testing_utils

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


if __name__ == '__main__':
    test_mock_reward()
