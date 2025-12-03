import numpy as np
from numpy import typing as npt
from itertools import product
from game_rules import RuleEvaluator
import matplotlib.pyplot as plt

"""
ULTIMATE GOAL: tell user the best move
    ie. if you roll [1, 5, 6, 2, 3, 3]
        Whether to 
          keep the 1, roll the rest OR 
          keep the 5, roll the rest, OR 
          keep 1 and 5, roll rerst
    
    Create point distribution for each move
         {roll_result: [distribution of points]}
            Only preserve first roll_result in dictioanry, everything else keeps iterating till final # of pts reached
           
"""
RULES = RuleEvaluator()
DICE_TOTAL = 6
ALL_DICE_COMBINATIONS = {i: [tuple(p) for p in product([1,2,3,4,5,6], repeat=i)] for i in range(7)}

def simulator(dice_rolls: npt.NDArray):
    """_summary_
    Args:
        dice_rolls (npt.NDArray): numpy array of dice rolls
        starter (bool, optional): Whether the call to simulator() is the first or not 
        the first. 
            if it's the first, each roll_result in roll_results will be a key in return dict
            else simulator() will keep iterating and return point values
        Defaults to True.

    Returns:
        Dict: {roll_result: [point distribution]}
    """
  
    initial_roll_results = RULES.check_roll(dice_rolls)
    if len(initial_roll_results) == 0:
        return 0 # has to override previous rolls
    scores_dict = {}
    memoer = {}
    
    # actual recursive fucntion
    def helper(remaining_dice: int, total_points: int): # calculate distribution of values for each roll_results
        # Memo efficiency!
        
        key = (remaining_dice, total_points)
        if key in memoer:
            return memoer[key]
        
        totals = []
        
        # no dice left (NOTE: ignoring 6 dice reset rule)
        if remaining_dice == 0:
            totals.append(total_points) 
            memoer[key] = totals
            return totals
        
        for new_roll in ALL_DICE_COMBINATIONS[remaining_dice]:
            new_roll = np.array(new_roll)
            roll_results = RULES.check_roll(new_roll)
            
            if len(roll_results) == 0: # BUST Rule
                totals.append(0)
                continue # skip the below

            for dice_value, num_dice, points in roll_results:
                next_remaining_dice = remaining_dice - num_dice
                next_points = total_points + points
                
                branch_totals = helper(next_remaining_dice, next_points) 
                totals.extend(branch_totals)
                
        memoer[key] = totals
        return totals
    
    
    for dice_value, num_dice, points in initial_roll_results:
        remaining_dice = DICE_TOTAL - num_dice
        point_distribution = helper(remaining_dice, points)
        
        scores_dict[(dice_value, num_dice, points)] = point_distribution
                
        
    return scores_dict





    
def low_score_remover(scores: dict, begin: bool = True):
    """Implements 350 and 1000 rule:
        if simulated_value: 0 points (begin=True) --> need to reach 1000 else 0 pts
        elif simulated_value: >1000 points (begin=False) -->  need to reach 350 pts else 0 pts
    """
    minimum = 1000 if begin else 350
    
    for key, value_list in scores.items():
        for i in range(len(value_list)): # iterate through list
            if value_list[i] < minimum:  
                value_list[i] = 0
        

def best_move(scores: dict):
    """Determine which move is the best by calculating statistics (tbd) from 
    each key's distribution in scores

    Args:
        scores (dict): _description_

    """
    ... 
    
    
    
def main():
    
    dice_rolls = np.random.randint(1, 7, 6) # NOTE: replace this with the inference result of the model
    begin = True
    
    scores_dict = simulator(dice_rolls)
    
    # low_score_remover(scores_dict, True)
    
    # best_move(scores_dict)
    
    

if __name__ == '__main__':
    main()