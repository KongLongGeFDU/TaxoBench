#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def calculate_f1(precision, recall):
    """ 
    Args:
        precision
        recall
    
    Returns:
        F1
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def main():
    nsr_list = [
    ]
    
    nsp_list = [
    ]
    min_len = min(len(nsr_list), len(nsp_list))
    f1_scores = []
    for i in range(min_len):
        nsr = nsr_list[i]
        nsp = nsp_list[i]
        f1 = calculate_f1(nsp, nsr) 
        f1_scores.append(f1)
    

    if f1_scores:
        print(f"F1: {[f'{f:.4f}' for f in f1_scores]}")



if __name__ == "__main__":
    main()

