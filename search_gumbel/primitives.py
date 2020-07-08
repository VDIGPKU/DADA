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
        
