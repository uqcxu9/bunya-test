import pickle

# 查看 actions_6.pkl 文件内容
print('=== actions_6.pkl 文件内容 ===')
with open('data/gpt-3-perception-reflection-1-100agents-240months/actions_6.pkl', 'rb') as f:
    actions = pickle.load(f)

print(f'数据类型: {type(actions)}')
print(f'总键数: {len(actions)}')
print(f'所有键: {list(actions.keys())}')
print()

# 显示前10个智能体的动作
print('前10个智能体的动作:')
for i in range(10):
    agent_id = str(i)
    if agent_id in actions:
        print(f'智能体{agent_id}: {actions[agent_id]}')
print()

# 显示规划者的动作
if 'p' in actions:
    print(f'规划者动作: {actions["p"]}')
print()

# 统计工作决策
work_decisions = []
consumption_decisions = []
for agent_id, action in actions.items():
    if agent_id != 'p':  # 排除规划者
        work_decisions.append(action[0])
        consumption_decisions.append(action[1])

print(f'工作决策统计:')
print(f'  工作人数: {sum(work_decisions)}')
print(f'  不工作人数: {len(work_decisions) - sum(work_decisions)}')
print(f'  工作率: {sum(work_decisions)/len(work_decisions)*100:.1f}%')
print()

print(f'消费决策统计:')
print(f'  平均消费率: {sum(consumption_decisions)/len(consumption_decisions):.2f}')
print(f'  最低消费率: {min(consumption_decisions)}')
print(f'  最高消费率: {max(consumption_decisions)}')
print()

# 显示所有智能体的完整动作
print('所有智能体的完整动作:')
for agent_id in sorted(actions.keys(), key=lambda x: int(x) if x.isdigit() else 999):
    if agent_id != 'p':
        work_status = "工作" if actions[agent_id][0] == 1 else "不工作"
        consumption_rate = actions[agent_id][1]
        print(f'智能体{agent_id}: {work_status}, 消费率={consumption_rate}')
    else:
        print(f'规划者: {actions[agent_id]}')






