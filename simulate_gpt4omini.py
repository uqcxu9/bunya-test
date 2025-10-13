from typing import Optional import argparse import fire import os import sys import ai_economist.foundation as foundation import numpy as np import matplotlib.pyplot as plt import yaml from time import time from collections import defaultdict import re from simulate_utils import * import pickle as pkl from itertools import product from dateutil.relativedelta import relativedelta CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml') with open(CONFIG_PATH, "r") as f: run_configuration = yaml.safe_load(f) env_config = run_configuration.get('env') agent_policy_cfg = run_configuration.get('agent_policy', {}) or {} USE_PERCEPTION = bool(agent_policy_cfg.get('use_perception', False)) USE_REFLECTION = bool(agent_policy_cfg.get('use_reflection', True)) def gpt_actions(env, obs, dialog_queue, dialog4ref_queue, gpt_path, gpt_error, total_cost): if not os.path.exists(gpt_path): os.makedirs(gpt_path) curr_rates = obs['p']['PeriodicBracketTax-curr_rates'] current_time = world_start_time + relativedelta(months=env.world.timestep) current_time = current_time.strftime('%Y.%m') for idx in range(env.num_agents): this_agent = env.get_agent(str(idx)) skill = this_agent.state['skill'] wealth = this_agent.inventory['Coin'] consumption = this_agent.consumption['Coin'] interest_rate = env.world.interest_rate[-1] price = env.world.price[-1] tax_paid = obs['p'][f'p{idx}']['PeriodicBracketTax-tax_paid'] lump_sum = obs['p'][f'p{idx}']['PeriodicBracketTax-lump_sum'] max_l = env._components_dict['SimpleLabor'].num_labor_hours name = this_agent.endogenous['name'] age = this_agent.endogenous['age'] city = this_agent.endogenous['city'] job = this_agent.endogenous['job'] offer = this_agent.endogenous['offer'] actions = env.dense_log['actions'] states = env.dense_log['states'] problem_prompt = f''' You're {name}, a {age}-year-old individual living in {city}. As with all Americans, a portion of your monthly income is taxed by the federal government. This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings. Now it's {current_time}. ''' if job == 'Unemployment': job_prompt = f''' In the previous month, you became unemployed and had no income. Now, you are invited to work as a(an) {offer} with monthly salary of ${skill*max_l:.2f}. ''' else: if skill >= states[-1][str(idx)]['skill']: job_prompt = f''' In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is increased compared to the last month due to the inflation of labor market. ''' else: job_prompt = f''' In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is decreased compared to the last month due to the deflation of labor market. ''' if (consumption <= 0) and (len(actions) > 0) and (actions[-1].get('SimpleConsumption', 0) > 0): consumption_prompt = f''' Besides, you had no consumption due to shortage of goods. ''' else: consumption_prompt = f''' Besides, your consumption was ${consumption:.2f}. ''' if env._components_dict['PeriodicBracketTax'].tax_model == 'us-federal-single-filer-2018-scaled': tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}. In this month, the government sets the brackets: {format_numbers(brackets)} and their corresponding rates: {format_numbers(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.''' else: tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}. In this month, according to the optimal taxation theory, Saez Tax, the brackets are not changed: {format_numbers(brackets)} but the government has updated corresponding rates: {format_percentages(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.''' if env.world.timestep == 0: price_prompt = f'''Meanwhile, in the consumption market, the average price of essential goods is now at ${price:.2f}.''' else: if price >= env.world.price[-2]: price_prompt = f'''Meanwhile, inflation has led to a price increase in the consumption market, with the average price of essential goods now at ${price:.2f}.''' else: price_prompt = f'''Meanwhile, deflation has led to a price decrease in the consumption market, with the average price of essential goods now at ${price:.2f}.''' job_prompt = prettify_document(job_prompt) # Optional perception stage: summarize key signals as a system message if USE_PERCEPTION: trend = 'increased' if (env.world.timestep == 0 or price >= env.world.price[-2]) else 'decreased' perception_msg = f"Perception: price has {trend} to ${price:.2f}, interest {interest_rate*100:.2f}%, last consumption ${consumption:.2f}, last tax ${tax_paid:.2f}, lump-sum ${lump_sum:.2f}." perception_msg = prettify_document(perception_msg) # print perception into terminal print(f"[agent {idx}] >>> system [perception]: {perception_msg}") dialog_queue[idx].append({'role': 'system', 'content': perception_msg}) dialog4ref_queue[idx].append({'role': 'system', 'content': perception_msg}) obs_prompt = f''' {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt} Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%. With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price? Respond with ONLY a single JSON object and NOTHING ELSE (no code fences, no explanations). The JSON must contain two keys: "work" and "consumption", both numbers between 0 and 1, using a step size of 0.02. Example: {{"work": 0.84, "consumption": 0.62}} ''' obs_prompt = prettify_document(obs_prompt) # print user prompt to terminal print(f"[agent {idx}] >>> user: {obs_prompt}") dialog_queue[idx].append({'role': 'user', 'content': obs_prompt}) dialog4ref_queue[idx].append({'role': 'user', 'content': obs_prompt}) def action_check(actions): if len(actions) != 2: return False else: return (actions[0] >= 0) & (actions[0] <= 1) & (actions[1] >= 0) & (actions[1] <= 1) if USE_REFLECTION and env.world.timestep%3 == 0 and env.world.timestep > 0: # merge a small slice of recent reflection context if available merged_prompts = [] for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue): merged = list(dialogs) # try to bring last reflection assistant message as system emphasis ref_assist = [d for d in list(dialog4ref)[-4:] if d.get('role') == 'assistant'] if len(ref_assist) > 0: merged.insert(0, {'role': 'system', 'content': f"Reflection summary: {ref_assist[-1]['content'][:400]}"}) merged_prompts.append(merged) results, cost = get_multiple_completion(merged_prompts) total_cost += cost else: results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog_queue]) total_cost += cost def _balanced_json_extract(text): # Strip code fences if present if '
' in text:
            # Try to capture content inside the first fenced block
            start = text.find('
') rest = text[start+3:] # skip optional language tag nl = rest.find('\n') if nl != -1: rest = rest[nl+1:] end_fence = rest.find('
')
            if end_fence != -1:
                text = rest[:end_fence]
        # Find first balanced {...}
        start_brace = text.find('{')
        if start_brace == -1:
            return None
        depth = 0
        for i in range(start_brace, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start_brace:i+1]
        return None

    def _parse_actions(content):
        import json
        snippet = _balanced_json_extract(content)
        if not snippet:
            return None
        try:
            obj = json.loads(snippet)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        work = obj.get('work')
        consumption = obj.get('consumption')
        if not isinstance(work, (int, float)) or not isinstance(consumption, (int, float)):
            return None
        # clamp to [0,1]
        work = max(0.0, min(1.0, float(work)))
        consumption = max(0.0, min(1.0, float(consumption)))
        # quantize to 0.02 steps
        def _quantize(x):
            steps = round(x / 0.02)
            return max(0.0, min(1.0, steps * 0.02))
        work = _quantize(work)
        consumption = _quantize(consumption)
        return [work, consumption]

    actions = {}
    for idx in range(env.num_agents):
        content = results[idx]
        # print assistant raw reply to terminal
        print(f"[agent {idx}] <<< assistant: {content}")
        extracted_actions = _parse_actions(content)
        if extracted_actions is None or not action_check(extracted_actions):
            extracted_actions = [1, 0.5]
            gpt_error += 1
        extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
        extracted_actions[1] /= 0.02
        actions[str(idx)] = extracted_actions
        dialog_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
        dialog4ref_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
    actions['p'] = [0]
    for idx, agent_dialog in enumerate(dialog_queue):
        with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
            for dialog in list(agent_dialog)[-2:]:
                f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
        
    if USE_REFLECTION and (env.world.timestep+1)%3 == 0:
        reflection_prompt = '''Given the previous quarter's economic environment, reflect on the labor, consumption, and financial markets, as well as their dynamics. What conclusions have you drawn?
        Your answer must be less than 200 words!'''
        reflection_prompt = prettify_document(reflection_prompt)
        for idx in range(env.num_agents):
            # dialog_queue[idx].append({'role': 'user', 'content': reflection_prompt})
            dialog4ref_queue[idx].append({'role': 'user', 'content': reflection_prompt})
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog4ref_queue], temperature=0, max_tokens=200)
        total_cost += cost
        for idx in range(env.num_agents):
            content = results[idx]
            # print reflection reply
            print(f"[agent {idx}] <<< assistant [reflection]: {content}")
            # dialog_queue[idx].append({'role': 'assistant', 'content': content})
            dialog4ref_queue[idx].append({'role': 'assistant', 'content': content})
        
        for idx, agent_dialog in enumerate(dialog4ref_queue):
             with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
                for dialog in list(agent_dialog)[-2:]:
                    f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
    return actions, gpt_error, total_cost

def complex_actions(env, obs, beta=0.1, gamma=0.1, h=1):

    def consumption_len(price, wealth, curr_income, last_income, interest_rate):
        c = (price/(1e-8+wealth+curr_income))**beta
        c = min(max(c//0.02, 0), 50)
        return c
    def consumption_cats(price, wealth, curr_income, last_income, interest_rate):
        h1 = h / (1 + interest_rate)
        g = curr_income/(last_income+1e-8) - 1
        d = wealth/(last_income+1e-8) - h1
        c = 1 + (d - h1*g)/(1 + g + 1e-8)
        c = min(max(c*curr_income/(wealth+curr_income+1e-8)//0.02, 0), 50)
        return c
    def work_income_wealth(price, wealth, curr_income, last_income, expected_income, interest_rate):
        return int(np.random.uniform() < (curr_income/(wealth*(1 + interest_rate)+1e-8))**gamma)
    
    consumption_funs = [consumption_len, consumption_cats]
    work_funs = [work_income_wealth]

    actions = {}
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        price = env.world.price[-1]
        wealth = this_agent.inventory['Coin']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        max_income = max_l * this_agent.state['skill']
        last_income = this_agent.income['Coin']
        expected_income = max_l * this_agent.state['expected skill']
        interest_rate = env.world.interest_rate[-1]
        if 'consumption_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['consumption_fun_idx'] = np.random.choice(range(len(consumption_funs)))
        if 'work_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['work_fun_idx'] = np.random.choice(range(len(work_funs)))
        work_fun = work_funs[this_agent.endogenous['work_fun_idx']]
        l = work_fun(price, wealth, max_income, last_income, expected_income, interest_rate)
        curr_income = l * max_income
        consumption_fun = consumption_funs[this_agent.endogenous['consumption_fun_idx']]
        c = consumption_fun(price, wealth, curr_income, last_income, interest_rate)
        actions[str(idx)] = [l, c]
    actions['p'] = [0]
    return actions
    

def main(policy_model='gpt', num_agents=100, episode_length=240, dialog_len=3, beta=0.1, gamma=0.1, h=1, max_price_inflation=0.1, max_wage_inflation=0.05):
    env_config['n_agents'] = num_agents
    env_config['episode_length'] = episode_length
    if policy_model == 'gpt':
        total_cost = 0
        env_config['flatten_masks'] = False
        env_config['flatten_observations'] = False
        env_config['components'][0]['SimpleLabor']['scale_obs'] = False
        env_config['components'][1]['PeriodicBracketTax']['scale_obs'] = False
        env_config['components'][3]['SimpleSaving']['scale_obs'] = False
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation
        
        gpt_error = 0
        from collections import deque
        dialog_queue = [deque(maxlen=dialog_len) for _ in range(env_config['n_agents'])]
        dialog4ref_queue = [deque(maxlen=7) for _ in range(env_config['n_agents'])]

    elif policy_model == 'complex':
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation

    t = time()
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    actions = {}
    if policy_model == 'complex':
        policy_model_save = f'{policy_model}-{beta}-{gamma}-{h}-{max_price_inflation}-{max_wage_inflation}'
    elif policy_model == 'gpt':
        perception_tag = 'perception' if USE_PERCEPTION else 'noperception'
        reflection_tag = 'reflection' if USE_REFLECTION else 'noreflection'
        policy_model_save = f'{policy_model}-{dialog_len}-{perception_tag}-{reflection_tag}-1'
    else:
        # 对于其他模型类型（如random），使用简单的命名
        policy_model_save = f'{policy_model}'
    
    policy_model_save = f'{policy_model_save}-{num_agents}agents-{episode_length}months'
    if not os.path.exists(f'{save_path}data/{policy_model_save}'):
        os.makedirs(f'{save_path}data/{policy_model_save}')
    if not os.path.exists(f'{save_path}figs/{policy_model_save}'):
        os.makedirs(f'{save_path}figs/{policy_model_save}')
    for epi in range(env.episode_length):
        if policy_model == 'gpt':
            actions, gpt_error, total_cost = gpt_actions(env, obs, dialog_queue, dialog4ref_queue, f'{save_path}data/{policy_model_save}/dialogs', gpt_error, total_cost)
        elif policy_model == 'complex':
            actions = complex_actions(env, obs, beta=beta, gamma=gamma, h=h)
        elif policy_model == 'random':
            # 随机策略：智能体随机选择动作
            actions = {}
            for idx in range(env.num_agents):
                # 随机选择劳动动作：0（不工作）或1（工作）
                work_action = np.random.randint(0, 2)
                # 随机选择消费动作：1到50之间的整数（对应0.02到1.0的消费率）
                consumption_action = np.random.randint(1, 51)
                actions[str(idx)] = [work_action, consumption_action]
            actions['p'] = [0]  # 规划者不采取行动
        else:
            raise ValueError(f"Unknown policy model: {policy_model}")
        
        obs, rew, done, info = env.step(actions)
        if (epi+1) % 3 == 0:
            print(f'step {epi+1} done, cost {time()-t:.1f}s')
            if policy_model == 'gpt':
                print(f'#errors: {gpt_error}, cost ${total_cost:.1f} so far')
            t = time()
        if (epi+1) % 6 == 0 or epi+1 == env.episode_length:
            with open(f'{save_path}data/{policy_model_save}/actions_{epi+1}.pkl', 'wb') as f:
                pkl.dump(actions, f)
            with open(f'{save_path}data/{policy_model_save}/obs_{epi+1}.pkl', 'wb') as f:
                pkl.dump(obs, f)
            with open(f'{save_path}data/{policy_model_save}/env_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env, f)
            if policy_model == 'gpt':
                with open(f'{save_path}data/{policy_model_save}/dialog_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog_queue, f)
                with open(f'{save_path}data/{policy_model_save}/dialog4ref_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog4ref_queue, f)
            with open(f'{save_path}data/{policy_model_save}/dense_log_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env.dense_log, f)
                
    with open(f'{save_path}data/{policy_model_save}/dense_log.pkl', 'wb') as f:
        pkl.dump(env.dense_log, f)
        
    if policy_model == 'gpt':
        print(f'#gpt errors: {gpt_error}')

if __name__ == "__main__":
    fire.Fire(main)
