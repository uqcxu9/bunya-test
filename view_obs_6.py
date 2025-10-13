import pickle

# 查看 obs_6.pkl 文件内容
print('=== obs_6.pkl 文件内容 ===')
with open('data/gpt-3-perception-reflection-1-100agents-240months/obs_6.pkl', 'rb') as f:
    obs = pickle.load(f)

print(f'数据类型: {type(obs)}')
print(f'总键数: {len(obs)}')
print(f'所有键: {list(obs.keys())}')
print()

# 显示智能体观测信息
print('=== 智能体观测信息 ===')
agent_keys = [k for k in obs.keys() if k != 'p']
print(f'智能体数量: {len(agent_keys)}')
print(f'智能体ID: {agent_keys[:10]}...')  # 显示前10个
print()

# 显示智能体0的详细观测
print('智能体0的详细观测:')
agent_0_obs = obs['0']
for key, value in agent_0_obs.items():
    print(f'  {key}: {value}')
print()

# 显示智能体1的详细观测
print('智能体1的详细观测:')
agent_1_obs = obs['1']
for key, value in agent_1_obs.items():
    print(f'  {key}: {value}')
print()

# 显示全局观测信息
print('=== 全局观测信息 ===')
global_obs = obs['p']
print(f'全局观测键数: {len(global_obs)}')
print('全局观测内容:')
for key, value in global_obs.items():
    print(f'  {key}: {value}')
print()

# 显示所有智能体的基本信息
print('=== 所有智能体的基本信息 ===')
for i in range(min(5, len(agent_keys))):  # 显示前5个智能体
    agent_id = str(i)
    if agent_id in obs:
        agent_obs = obs[agent_id]
        skill = agent_obs.get('SimpleLabor-skill', 'N/A')
        age = agent_obs.get('SimpleLabor-age', 'N/A')
        wealth = agent_obs.get('SimpleSaving-wealth', 'N/A')
        price = agent_obs.get('SimpleConsumption-price', 'N/A')
        print(f'智能体{agent_id}: 技能={skill}, 年龄={age}, 财富={wealth}, 价格={price}')
print()

# 显示宏观经济指标
print('=== 宏观经济指标 ===')
world_metrics = ['world-normalized_per_capita_productivity', 'world-equality', 
                 'world-normalized_per_capita_cum_pretax_income', 'world-normalized_per_capita_consumption']
for metric in world_metrics:
    if metric in global_obs:
        print(f'{metric}: {global_obs[metric]}')
print()

# 显示税收信息
print('=== 税收信息 ===')
tax_metrics = ['PeriodicBracketTax-curr_rates', 'PeriodicBracketTax-is_tax_day', 
               'PeriodicBracketTax-tax_phase']
for metric in tax_metrics:
    if metric in global_obs:
        print(f'{metric}: {global_obs[metric]}')
print()

# 显示时间信息
print('=== 时间信息 ===')
if 'time' in global_obs:
    print(f'当前时间: {global_obs["time"]}')
if 'time' in obs['0']:
    print(f'智能体时间: {obs["0"]["time"]}')





