import random
import json

map_id = ['c15', 'c20', 'c25']
unloading_dict = {}
for m in map_id:
    for i in range(0, 10):
        name = m+'_'+str(i)
        unloading = []
        for j in range(0, 300):
            unloading.append(random.uniform(-6, 6))
            
        unloading_dict[name] = unloading
with open("unloading_pos.txt", 'w') as outfile:
    json.dump(unloading_dict, outfile)

