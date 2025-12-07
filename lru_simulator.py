from functools import lru_cache
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from game_rules import RuleEvaluator
import matplotlib.pyplot as plt
import plotly.graph_objects as go

RULES = RuleEvaluator()
DICE_TOTAL = 6

### OPTIMIZATION ###
def all_histograms(n):
    """Return all count histograms of n dice."""
    result = []
    for combo in combinations_with_replacement([1,2,3,4,5,6], n):
        counts = [0] * 6
        for v in combo:
            counts[v - 1] += 1
        result.append(tuple(counts))
    return result

ALL_HISTS = {i: all_histograms(i) for i in range(7)}

@lru_cache(maxsize=None)
def helper(remaining_dice: int, total_points: int) -> tuple[int]:
    if remaining_dice == 0:
        return (int(total_points),)
    
    totals = []
    
    for counts in ALL_HISTS[remaining_dice]:
        roll_results = RULES.check_roll_from_counts(list(counts))
        
        if len(roll_results) == 0:  # Bust
            totals.append(0)
            continue
        
        for dice_value, num_used, pts in roll_results:
            next_remaining = remaining_dice - num_used
            next_total = total_points + pts
            branch = helper(next_remaining, next_total)
            totals.extend([int(x) for x in branch])
    
    return tuple(totals)

### SIMULATOR ###
# TODO: Simulate SOLELY the next diec roll
# Later TODO: Player can choose to stop rolling

def simulator(dice_rolls: np.ndarray) -> dict | int:
    initial_rolls = RULES.check_roll(dice_rolls)
    
    if len(initial_rolls) == 0:
        return 0
    
    scores_dict = {}
    
    for dice_value, num_dice, points in initial_rolls:
        remaining = DICE_TOTAL - num_dice
        outcomes = helper(remaining, points)
        scores_dict[(dice_value, num_dice, points)] = list(outcomes)
    
    return scores_dict

def low_score_remover(scores: dict, begin: bool = True):
    """Implements 350 and 1000 rule:
        if simulated_value: 0 points (begin=True) --> need to reach 1000 else 0 pts
        elif simulated_value: >1000 points (begin=False) -->  need to reach 350 pts else 0 pts
    """
    minimum = 1000 if begin else 350
    
    for _, value_list in scores.items():
        for i in range(len(value_list)): # iterate through list
            if value_list[i] < minimum:  
                value_list[i] = 0
    return scores
    

### DISTRIBUTION PLOTTER ###
def create_distribution(scores: dict) -> None:
    """
    every image is one move

    """
    for key in scores.keys():
        values = np.array(scores[key], dtype=int)
        
        unique_scores, counts = np.unique(values, return_counts=True)
        percentages = (counts / len(values)) * 100
        
        #### Plotly ####

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=unique_scores,
            y=percentages,
            name = "Point Distribution",
            marker_color = 'darkblue',
            hovertemplate='%{y:.3f}% of distribution<extra>Score: %{x}</extra>'
        ))
        
        fig.update_layout(
            title=f"Distribution for Move: {key}",
            xaxis_title = "Score",
            yaxis_title = "Frequency (%)",
            template = "plotly_white",
            bargap = 0,
            hovermode = "closest"
        )
        
        fig.write_image(f"Distribution_{key}.png")
        fig.show()
    
    #### Matplotlib ####
    
    # plt.hist(scores[key])
    # plt.title(f"Dice Roll: {key}")
    # plt.xlabel("Point Distribution")
    # plt.ylabel("Frequency")
    # plt.savefig(f"Sim: {key}.png")
    # plt.show()


# TBD
def best_move(scores: dict):
    """Determine which move is the best by calculating statistics (tbd) from 
    each key's distribution in scores

    Args:
        scores (dict): _description_

    """
    means = np.array([]) # array of each moves means
    medians = np.array([]) # arrayt of each moves medians
    
    scores_df = pd.DataFrame(scores)
    
    for key in scores.keys():
        values = np.array(scores[key], dtype=int)
        means = np.append(means, np.mean(values))
        medians = np.append(medians, np.median(values))
        
    
    return means, medians    


def main():
    np.random.seed(42)
    # dice_rolls = np.random.randint(1,7,6)
    
    dice_rolls = np.array([1, 1, 1, 1, 1, 1])
    
    print(f"Dice rolled: {dice_rolls}")
    
    scores_dict = simulator(dice_rolls)
    
    print("Initial possible move distributions:")
    for k,v in scores_dict.items(): # type: ignore
        print(f"{k} --> {v[:10]}...")
    
    scores_dict = low_score_remover(scores_dict, True) # type: ignore
    
    create_distribution(scores_dict) # type: ignore

if __name__=="__main__":
    main()