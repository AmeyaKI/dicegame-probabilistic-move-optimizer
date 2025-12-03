import numpy as np
from numpy import typing as npt

# EDIT TO MATCH SIMULATOR NOT THEORETICAL
class RuleEvaluator():
    """
    Rough List of Rules:
    3 1s --> 1000
    3 of any other # --> 100 * #

    1 1 --> 100
    1 5 --> 50
    """
    DICE_VALUES = np.array([1, 2, 3, 4, 5, 6])
    
    def check_roll(self, dice_rolls: npt.NDArray, combined_branch=True):
        """Checks array of dice rolls to determine if the user rolled any dice 
        resulting in points and if so, how many points and which combinations 
        result in points

        Args:
            dice_rolls (npt.NDArray): numpy array of values of dice rolled
            combined_branch (bool, optional): _description_. Defaults to True.
            
        Returns:
            Tuple(int) (dice_value, num_dice, points): 
                dice_value: number on dice face
                num_dice: number of dice
                points: total number of points
        """
                
        roll_results = []
        for dice_value in RuleEvaluator.DICE_VALUES:
            num_dice = np.count_nonzero(dice_rolls == dice_value)
            
            while num_dice >= 3:
                if dice_value == 1: # 3 1s --> 1000
                    result = [dice_value, 3, 1000]
                    roll_results.append(result)
                else:# 3 num --> 100 * num
                    result = [dice_value, 3, dice_value * 100]
                    roll_results.append(result)

                num_dice -= 3

            if num_dice > 0:
                if dice_value == 1: # 1 1 --> 100
                    result = [dice_value, num_dice, num_dice * 100]
                    roll_results.append(result)
                elif dice_value == 5: # 1 5 --> 50
                    result = [dice_value, num_dice, num_dice * 50]
                    roll_results.append(result)
        
        if combined_branch and roll_results:
            # CHANGE TO ALL POSSIBLE COMBINATIONS 
                # if you roll [1, 3, 1000], [5, 1, 50], [5, 1, 50]
                # combined  results: 
                    # [1, 3, 1000] + [5, 1, 50] kept, reroll rest
                    # [5, 1, 50] + [5, 1, 50] kept, reroll rest
                    # etc
            total_points = sum([points for _, _, points in roll_results])
            total_dice = sum([num_dice for _, num_dice, _ in roll_results])
            roll_results.append(['all', total_dice, total_points])
        return roll_results
  
            
    