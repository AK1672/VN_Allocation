import pickle
import random

def generate_vnr(filename, num_vms=3):
    vnr = {
        "vms": [
            {
                "id": f"vm{i}",
                "cpu": random.randint(10, 30),       # CPU requirement
                "memory": random.randint(100, 500), # Memory requirement
                "storage": random.randint(100, 1000) # Storage requirement
            }
            for i in range(num_vms)
        ],
        "links": [
            {
                "src": f"vm{i}",
                "dst": f"vm{j}",
                "bandwidth": random.randint(10, 50) # Bandwidth requirement
            }
            for i in range(num_vms) for j in range(i + 1, num_vms)
        ]
    }

    with open(filename, 'wb') as f:
        pickle.dump(vnr, f)

    print(f"VNR saved to {filename}")


for i in range(10): 
    generate_vnr(f"vnr_{i}.pkl", num_vms=random.randint(2, 5))