import re
import json 

def get_skill_changepoints(state_dict):
    change_points = []

    for i in range(1, len(state_dict)):
        previous = state_dict[i - 1]
        current = state_dict[i]
        
        # Check each item in the current inventory
        for item, current_value in current.items():
            previous_value = previous.get(item, 0)
            if current_value > previous_value:
                change_points.append((item, i))
    
    return change_points

def get_skills(change_points):
    lines = []     
    prev_total = 0 
    group_sum = 0  
    prev_key = None

    for key, value in change_points:
        if key == prev_key:
            count = value - prev_total
        else:
            count = value - prev_total
            
        for _ in range(count):
            lines.append(key)
        prev_total = value
        prev_key = key

    final_string = "\n".join(lines)
    return final_string

def transform_string(s):
    return re.sub(r'.*(log|planks).*', r'\1', s, flags=re.IGNORECASE)

def get_unique_dict_items(data):
    item_names = set()
    for inv_dict in data:
        for entry_ in inv_dict.values():
            item_names.add(transform_string(entry_['type']))
    
    item_names.remove('none')
    return dict.fromkeys(item_names, 0)

def minestudio_inv_to_skills(minestudio_inv):
    better_dict = get_unique_dict_items(minestudio_inv)


    all_better_dicts = []

    #Use this to filter out noise like picking up random stuff
    allowed_items = [
        'planks',
        'log',
        'crafting_table',
        'stick',
        'wooden_pickaxe'
    ]

    for inv_dict in minestudio_inv:
        for _, inv_info in inv_dict.items():
            item_name = transform_string(inv_info['type'])
            item_qty = inv_info['quantity']
            if item_name in allowed_items:
                better_dict[item_name] = item_qty

            if item_name not in allowed_items and item_name != 'none':
                print("Missed: ", item_name)

        all_better_dicts.append(better_dict.copy())

    changepoints = get_skill_changepoints(all_better_dicts)

    skills = get_skills(changepoints)

    return skills 

def check_episode_done(inv_dict, item, qty):
    for _, inv_details in inv_dict.items():
        item_name = transform_string(inv_details['type'])
        item_qty = inv_details['quantity']
        if item_name == item and item_qty >= qty:
            return True
        
    return False


if __name__ == '__main__':

    with open("inv_data.json", "rb") as file:
        loaded_data = json.load(file)

    print(check_episode_done(loaded_data[-1], "wooden_pickaxe", 1))