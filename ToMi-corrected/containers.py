containers = [
    "box",
    "pantry",
    "bathtub",
    "envelope",
    "drawer",
    "bottle",
    "cupboard",
    "basket",
    "crate",
    "suitcase",
    "bucket",
    "container",
    "treasure_chest",
]

colors = ['green', 'blue', 'red']

containers = ['_'.join([color, container])
              for container in containers
              for color in colors]

for container in containers:
    print('"'+container+'"'+",")