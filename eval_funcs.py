import torch as th
from scipy.stats import kendalltau
from scipy.special import comb


def kendall_tau(edu_order):
    num_edus = edu_order.shape[0]
    true_order = th.arange(num_edus) + 1
    return kendalltau(edu_order, true_order)[0]
        
def blocked_kendall_tau(edu_order):
    num_edus = edu_order.shape[0]
    true_order = th.arange(num_edus) + 1
    if (edu_order == true_order).all():
        return 1.0
    block_edu_order = merge_blocks(edu_order)
    num_blocked_edus = len(block_edu_order)
    block_true_order = th.arange(num_blocked_edus) + 1
    blocked_kendall = 1 + (kendalltau(block_edu_order, block_true_order)[0] - 1) * comb(num_blocked_edus, 2) / comb(num_edus, 2)
    
    return blocked_kendall
    
def position_match(edu_order):
    num_edus = edu_order.shape[0]
    true_order = th.arange(num_edus) + 1
    return (true_order == edu_order.cpu()).sum().float() / num_edus
        
def perfect_match(edu_order):
    true_order = th.arange(edu_order.shape[0]) + 1
    return (true_order == edu_order.cpu()).all().float()

def merge_blocks(edu_order):
    block_edu_order = []
    curr_block = []
    for entry in edu_order:
        if curr_block == [] or entry == curr_block[-1] + 1:
            curr_block.append(entry)
        else:
            block_edu_order.append(curr_block[0])
            curr_block = [entry]
    block_edu_order.append(curr_block[0])
    return block_edu_order

def display_results(total_loss, 
                    logprob_acc, 
                    count, 
                    kendalls, 
                    block_kendalls, 
                    position_match_score, 
                    perfect_match_score, 
                    uas_acc, 
                    las_acc):
    print("epoch ended, total loss is ", total_loss)
    print("Dev loss: ", logprob_acc)
    print("Kendall's tau: ", kendalls / count)
    print("Blocked Kendall's tau: ", block_kendalls / count)
    print("Position match score: ", position_match_score / count)
    print("Perfect match score: ", perfect_match_score / count)
    print("UAS: ", float(uas_acc) / count)
    print("LAS: ", float(las_acc) / count)
