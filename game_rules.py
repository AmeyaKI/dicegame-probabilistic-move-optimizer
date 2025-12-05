import numpy as np
from numpy import typing as npt
from itertools import combinations
class RuleEvaluator():
    """
    Rough List of Rules:
    3 1s --> 1000
    3 of any other # --> 100 * #

    1 1 --> 100
    1 5 --> 50
    """
    DICE_VALUES = np.array([1, 2, 3, 4, 5, 6])
    
    def check_roll(self, dice_rolls: npt.NDArray):
        # """Checks array of dice rolls to determine if the user rolled any dice 
        # resulting in points and if so, how many points and which combinations 
        # result in points

        # Args:
        #     dice_rolls (npt.NDArray): numpy array of values of dice rolled
        #     combined_branch (bool, optional): _description_. Defaults to True.
            
        # Returns:
        #     Tuple(int) (dice_value, num_dice, points): 
        #         dice_value: number on dice face
        #         num_dice: number of dice
        #         points: total number of points
        # """
        counts = [np.count_nonzero(dice_rolls==v) for v in self.DICE_VALUES] # [1, 1, 1, 1, 1, 1] --> [6, 0, 0, 0, 0, 0]
        return self.check_roll_from_counts(counts)
    
    def check_roll_from_counts(self, counts: list[int]) -> list[tuple[int,int,int]]:
        moves = []
        
        # Triples
        for index in range(6):
            num_of_dice = counts[index]
            while num_of_dice >= 3:
                val = index + 1
                if val == 1:  # 1s
                    moves.append((1, 3, 1000))
                else:
                    moves.append((val, 3, val * 100))    
                num_of_dice -= 3   
            # if counts[index] >= 3:
            #     val = index + 1
            #     if val == 1:  # 1s
            #         moves.append((1, 3, 1000))
            #     else:
            #         moves.append((val, 3, val * 100))
                    
        # Singles (1s and 5s)
        for index in [0, 4]:  # 1 and 5
            val = index + 1
            num_of_dice = counts[index]
            
            if num_of_dice > 0:
                for k in range(1, num_of_dice + 1):
                    if k % 3 != 0:
                        points = k * 100 if val == 1 else k * 50
                        moves.append((val, k, points))
            
        return moves
        
        
        
        
        
####### PREVIOUS IMPLEMENTATION OF check_roll() #######        
        # roll_results = []
        # for dice_value in RuleEvaluator.DICE_VALUES:
        #     num_dice = np.count_nonzero(dice_rolls == dice_value)
            
        #     while num_dice >= 3:
        #         if dice_value == 1: # 3 1s --> 1000
        #             result = [dice_value, 3, 1000]
        #             roll_results.append(result)
        #         else:# 3 num --> 100 * num
        #             result = [dice_value, 3, dice_value * 100]
        #             roll_results.append(result)

        #         num_dice -= 3

        #     if num_dice > 0:
        #         if dice_value == 1: # 1 1 --> 100
        #             result = [dice_value, num_dice, num_dice * 100]
        #             roll_results.append(result)
        #         elif dice_value == 5: # 1 5 --> 50
        #             result = [dice_value, num_dice, num_dice * 50]
        #             roll_results.append(result)
        
        # combined_branch = False
        
        # #### KEEP THIS OFF FOR NOW
        # # if roll_results and combined_branch: 
        # #     # Account for all combiantions of num_dice
        # #     for i in range(len(roll_results)):
        # #         for dice_value, dice_num, points in roll_results[i]:
        # #             if dice_num % 3 != 0 and dice_num > 1 and type(dice_num) == int: # not a triple and not 1
        # #                 points_per_dice = points / dice_num
        # #                 iterator = dice_num
        # #                 while iterator > 0:
        # #                     roll_results.append([dice_value, dice_num - 1, points_per_dice])
        # #                     iterator -= 1
            
        # #     # Account for all possible combinations
        # #     combos = list(combinations(roll_results, 2))
        # #     for i in range(len(combos)):
        # #         total_points = sum([points for _, _, points in combos[i]])
        # #         total_num_dice = sum([num_dice for _, num_dice, _ in combos[i]])
        # #         total_dice_values = [dice_value for dice_value, _, _ in combos[i]]
        # #         roll_results.append([total_dice_values, total_num_dice, total_points])
        # #         # if you roll [1, 3, 1000], [5, 1, 50], [5, 1, 50]
        # #         # combined  results: 
        # #             # [1, 3, 1000] + [5, 1, 50] kept, reroll rest
        # #             # [5, 1, 50] + [5, 1, 50] kept, reroll rest
        # #             # etc
    
        # return roll_results
  
            
    