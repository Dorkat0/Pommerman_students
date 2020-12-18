# def convert_agents(board, location, can_kick, blast_strength, ammo, own_agent):
#     """ creates two 'clones' of the actual agents """
#     ret = []
#     # agent board ids are 10 and 11 in two-player games
#     for aid in [10, 11]:
#         locations = np.where(board == aid)
#
#         #TBE differentiate between enemy and our algo
#         posX, posY = location
#         if locations[0] == posX and locations[1] == posY:       #TBE enemy
#             agt = heuristic_agent.HeuristicAgent()
#         else:
#             agt = group05_agent.Group05Agent()                  #TBE our algo
#
#         agt.init_agent(aid, constants.GameType.FFA)
#         if len(locations[0]) != 0:
#             agt.set_start_position((locations[0][0], locations[1][0]))
#         else:
#             agt.set_start_position((0, 0))
#             agt.is_alive = False
#
#         if locations[0] == posX and locations[1] == posY:  # TBE enemy
#             agt.reset(is_alive=agt.is_alive,
#                       ammo=own_agent.get_enemy_items()["ammo"],
#                       blast_strength=own_agent.get_enemy_items()["blast_strength"],
#                       can_kick=own_agent.get_enemy_items()["can_kick"])
#         else:
#             agt.reset(is_alive=agt.is_alive,    #TBE our agent
#                       ammo=ammo,
#                       blast_strength=blast_strength,
#                       can_kick=can_kick)
#
#         agt.agent_id = aid - 10
#         ret.append(agt)
#     return ret

# def get_possible_movements(obs):
#     possible_movements = [0, 1, 2, 3, 4, 5]
#     position = obs['position']
#     barriers = [1, 2, 3, 4]
#
#     # already a bomb on the place        #TODO two bombs are not possible, i guess.
#     if (3 == obs['board'][position[0]][position[1]]):
#         possible_movements.remove(5)
#
#     # TODO check if bomb is blaced but the enemy of over the bomb
#
#     # boarder or barrier
#     if obs['position'][0] > 9 or obs['board'][position[0] + 1][position[1]] in barriers:  # boarder or barrier
#         possible_movements.remove(1)  # don´t go further up
#     elif obs['position'][0] == 0 or obs['board'][position[0] - 1][position[1]] in barriers:
#         possible_movements.remove(2)  # don´t go further down
#     if obs['position'][1] > 9 or obs['board'][position[0]][position[1] + 1] in barriers:
#         possible_movements.remove(4)  # don´t go further right
#     elif obs['position'][1] == 0 or obs['board'][position[0]][position[1] - 1] in barriers:
#         possible_movements.remove(3)  # don´t go further left
#
#     return possible_movements


# def get_enemy_bombs(self):

# def get_new_bombs(self, obs):
#     """ converts bomb matrices into bomb object list that can be fed to the forward model """
#     ret = []
#     locations = np.where(obs["bomb_life"] == 9)
#     for r, c in zip(locations[0], locations[1]):
#         is_enemy_bomb = False
#         if obs["board"][(r, c)] == self.enemy_value:
#             is_enemy_bomb = True
#         ret.append(
#             {'is_enemy_bomb': is_enemy_bomb, 'position': (r, c), 'blast_strength': int(obs["bomb_blast_strength"][(r, c)]), 'bomb_life': int(obs["bomb_life"][(r, c)]),
#              'moving_direction': None})
#
#     for i in ret:
#         bomber = None
#         if i['is_enemy_bomb']:
#             bomber = characters.Bomber(agent_id=self.enemy_agent_id)
#         else:
#             bomber = characters.Bomber(agent_id=self.agent_id)
#         #bomber = characters.Bomber()  # dummy bomber is used here instead of the actual agent
#         self.bomb_list.append(
#             characters.Bomb(bomber, i['position'], i['bomb_life'], i['blast_strength'],
#                             i['moving_direction']))

# def update_enemy_items(self, obs, old_obs):
#     #todo update current enemy ammo
#     #if self.old_enemy_pos is not None and obs["board"][self.old_enemy_pos[0][self.old_enemy_pos[2]] == 3:
#
#     posY, posX = group05_utils.get_enemy_position(obs)
#
#     board_status = old_obs["board"][posY][posX]
#     if board_status == 6:
#         self.enemy_items["ammo"] += 1
#     if board_status == 7:
#         self.enemy_items["blast_strength"] += 1
#     if board_status == 8:
#         self.enemy_items["can_kick"] = True
