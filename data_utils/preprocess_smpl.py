import numpy as np
import pickle
import sys

output_path = './model_lsp.pkl'

if __name__ == '__main__':
  src_path = '../../smpl_toy/res/neutral_smpl_with_cocoplus_reg.pkl'
  with open(src_path, 'rb') as f:
    src_data = pickle.load(f, encoding="latin1")
  model = {
    'J_regressor': src_data['J_regressor'],
    'joint_regressor': src_data['cocoplus_regressor'][:14],
    'weights': np.array(src_data['weights']),
    'posedirs': np.array(src_data['posedirs']),
    'v_template': np.array(src_data['v_template']),
    'shapedirs': np.array(src_data['shapedirs']),
    'f': np.array(src_data['f']),
    'kintree_table': src_data['kintree_table'],
  }
  with open(output_path, 'wb') as f:
    pickle.dump(model, f)
