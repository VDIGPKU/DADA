__all__ = ['sub_policies']
ops_names = [
    'ShearX', 
    'ShearY', 
    'TranslateX', 
    'TranslateY', 
    'Rotate', 
    'AutoContrast', 
    'Invert', 
    'Equalize', 
    'Solarize', 
    'Posterize', 
    'Contrast', 
    'Color', 
    'Brightness', 
    'Sharpness', 
    'Cutout', 
        ]
K = 2
sub_policies = []

def dfs(index=0, sub_policy=[], depth=0, sub_policies=[]):
    if depth == K:
        sub_policies += [tuple(sub_policy)]
        return
    for i, ops_name in enumerate(ops_names):
        if i < index:
            continue
        dfs(i+1, sub_policy + [ops_name], depth+1, sub_policies)

dfs(index=0, sub_policy=[], depth=0, sub_policies=sub_policies)
        

import numpy as np
for seed in range(0, 20):
    np.random.seed(seed)
    index = np.random.permutation(np.arange(len(sub_policies)))[0:25]
    pm = np.random.uniform(0,1, size=(25, 4))
    genotype = []
    for i in range(25):
        genotype += [((sub_policies[index[i]][0], pm[i][0], pm[i][1]), (sub_policies[index[i]][1], pm[i][2], pm[i][3]))]
    print("random_%d =" % seed , genotype)
