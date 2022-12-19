from taxi_matching import World

######## small input #########
customers = [(2, 2), (5, 5), (8, 6), (3, 7), (10, 9), (11, 7)]
dests = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
taxis = [(1, 9), (2, 9), (8, 8), (9, 3), (11, 2), (6, 1), (4, 2), (6, 1)]

world = World(world_size=(15, 15), random=False)
world.taxis(taxis)
for c, d in zip(customers, dests):
    world.add(c, d)

print("** small input **")
print(world.genetic(max_generation=20, selected_num=4))

######## random small input #######
print("** random small input **")
world = World(world_size=(20, 20), random=True, num_taxis=10, num_customers=7)
print("complete solution : ", world.complete())
print("greedy solution : ", world.greedy())
print("genetic solution : ", world.genetic(max_generation=20, selected_num=4))

######## random big input #######
print("** random big input **")
world = World(world_size=(50, 50), random=True, num_taxis=50, num_customers=40)
print("greedy solution : ", world.greedy())
print("genetic solution : ", world.genetic(max_generation=200, selected_num=10))
