import pickle

def view_agent0_obs(filename):
    """查看指定obs文件中Agent 0的观测信息"""
    print(f'=== {filename} 中 Agent 0 的观测信息 ===')
    
    try:
        with open(f'data/gpt-3-perception-reflection-1-100agents-240months/{filename}', 'rb') as f:
            obs = pickle.load(f)
        
        print(f'数据类型: {type(obs)}')
        print(f'总键数: {len(obs)}')
        print()
        
        # 检查Agent 0是否存在
        if '0' in obs:
            agent_0_obs = obs['0']
            print('Agent 0 的详细观测信息:')
            print('=' * 50)
            
            for key, value in agent_0_obs.items():
                print(f'{key}: {value}')
            
            print()
            print('Agent 0 的关键信息摘要:')
            print('=' * 50)
            
            # 显示关键信息
            key_info = {
                '时间': agent_0_obs.get('time', 'N/A'),
                '技能': agent_0_obs.get('SimpleLabor-skill', 'N/A'),
                '年龄': agent_0_obs.get('SimpleLabor-age', 'N/A'),
                '财富': agent_0_obs.get('SimpleSaving-wealth', 'N/A'),
                '价格': agent_0_obs.get('SimpleConsumption-price', 'N/A'),
                '消费率': agent_0_obs.get('SimpleConsumption-Consumption Rate', 'N/A'),
                '储蓄回报': agent_0_obs.get('SimpleSaving-Saving Return', 'N/A'),
                '是否税收日': agent_0_obs.get('PeriodicBracketTax-is_tax_day', 'N/A'),
                '税收阶段': agent_0_obs.get('PeriodicBracketTax-tax_phase', 'N/A'),
                '边际税率': agent_0_obs.get('PeriodicBracketTax-marginal_rate', 'N/A')
            }
            
            for key, value in key_info.items():
                print(f'{key}: {value}')
                
        else:
            print('错误: Agent 0 不存在于该文件中')
            
    except FileNotFoundError:
        print(f'错误: 文件 {filename} 不存在')
    except Exception as e:
        print(f'错误: {e}')

# 查看 obs_5.pkl 中的 Agent 0
print("查看 obs_5.pkl 中的 Agent 0:")
view_agent0_obs('obs_5.pkl')

print("\n" + "="*80 + "\n")

# 查看 obs_7.pkl 中的 Agent 0
print("查看 obs_7.pkl 中的 Agent 0:")
view_agent0_obs('obs_7.pkl')





