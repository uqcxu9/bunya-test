import pickle

# 查看 actions_6.pkl
print('=== actions_6.pkl 内容 ===')
with open('data/gpt-3-perception-reflection-1-100agents-240months/actions_6.pkl', 'rb') as f:
    actions = pickle.load(f)
print(f'类型: {type(actions)}')
print(f'键: {list(actions.keys())[:10]}...')  # 显示前10个键
print(f'智能体0的动作: {actions["0"]}')
print(f'智能体1的动作: {actions["1"]}')
print(f'智能体2的动作: {actions["2"]}')
print(f'规划者动作: {actions["p"]}')
print(f'总智能体数: {len([k for k in actions.keys() if k != "p"])}')

print('\n=== obs_6.pkl 内容 ===')
with open('data/gpt-3-perception-reflection-1-100agents-240months/obs_6.pkl', 'rb') as f:
    obs = pickle.load(f)
print(f'类型: {type(obs)}')
print(f'键: {list(obs.keys())}')
print(f'智能体观测键: {list(obs["0"].keys())}')
print(f'全局观测键: {list(obs["p"].keys())}')
print(f'智能体0的观测示例:')
for key, value in list(obs['0'].items())[:5]:
    print(f'  {key}: {value}')
print(f'全局观测示例:')
for key, value in list(obs['p'].items())[:5]:
    print(f'  {key}: {value}')

print('\n=== dialog_6.pkl 内容 ===')
with open('data/gpt-3-perception-reflection-1-100agents-240months/dialog_6.pkl', 'rb') as f:
    dialog = pickle.load(f)
print(f'类型: {type(dialog)}')
print(f'对话队列数量: {len(dialog)}')
print(f'智能体0的对话历史:')
for i, msg in enumerate(dialog[0]):
    print(f'  消息{i}: {msg}')
print(f'智能体1的对话历史:')
for i, msg in enumerate(dialog[1]):
    print(f'  消息{i}: {msg}')

print('\n=== dense_log_6.pkl 内容 ===')
with open('data/gpt-3-perception-reflection-1-100agents-240months/dense_log_6.pkl', 'rb') as f:
    dense_log = pickle.load(f)
print(f'类型: {type(dense_log)}')
print(f'键: {list(dense_log.keys())}')
print(f'actions 长度: {len(dense_log["actions"])}')
print(f'states 长度: {len(dense_log["states"])}')
print(f'前几个时间步的actions:')
for i in range(min(3, len(dense_log['actions']))):
    print(f'  时间步{i}: {dense_log["actions"][i]}')
print(f'前几个时间步的states示例:')
for i in range(min(2, len(dense_log['states']))):
    print(f'  时间步{i}的智能体0状态: {dense_log["states"][i]["0"]}')






