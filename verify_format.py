# verify_format.py
import pickle

with open("data/gpt-3-noperception-reflection-1-3agents-3months/dense_log.pkl", "rb") as f:
    log = pickle.load(f)

print("✅ Checking action format...")
for month in range(3):
    for agent_id in ['0', '1', '2']:
        action = log['actions'][month][agent_id]
        print(f"Month {month}, Agent {agent_id}: {action}")
        
        # 验证格式
        assert isinstance(action, dict), "❌ Not a dict!"
        assert 'SimpleLabor' in action, "❌ Missing SimpleLabor!"
        assert 'SimpleConsumption' in action, "❌ Missing SimpleConsumption!"
        assert isinstance(action['SimpleLabor'], (int, np.integer)), "❌ SimpleLabor not int!"
        assert isinstance(action['SimpleConsumption'], (int, np.integer)), "❌ SimpleConsumption not int!"

print("\n✅ All format checks passed!")