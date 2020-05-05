# -*- coding: UTF-8 -*-
import unittest

import matplotlib.pyplot as plt
from BioNAS.KFunctions.K_func import *
from BioNAS.resources.eclip import cnnModel
from BioNAS.utils import motif


class MotifKfunction_Test(unittest.TestCase):

    def test_MKF(self):
        mkf = Motif_K_function(temperature=0.1, Lambda_regularizer=0.01)
        mkf.knowledge_encoder(['RBFOX2_v1', 'RBFOX2_v2', 'RBFOX2_v3'], '../resources/rbp_motif/human_rbp_pwm.txt', True)

        model = cnnModel.build_model()
        print(mkf(model, None))
        model.load_weights('../resources/eclip/bestmodel.h5')
        print(mkf(model, None))

        score_dict, weight_dict = mkf.get_matched_model_weights()
        motif.draw_dnalogo_matplot(weight_dict['RBFOX2_v1'])
        plt.close()
        motif.draw_dnalogo_matplot(mkf.W_knowledge['RBFOX2_v1'])
        plt.close()


if __name__ == "__main__":
    unittest.main()
