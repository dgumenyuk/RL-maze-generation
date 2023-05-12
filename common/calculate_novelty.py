

def intersect(tc1, tc2):
    """
    Compute the intersection of two sets (two test cases)

    Args:
        state1: the first element to compare
        state2: the second element to compare

    Returns:
        The list of similar elements in the two test cases 
    """
    intersection = []
    tc_size  = min(len(tc1), len(tc2))
    for i in range(tc_size):
        if tc1[i][0] == tc2[i][0]:
            if (abs(tc1[i][1] - tc2[i][1]) <= 2) and (abs(tc1[i][2] - tc2[i][2]) <= 2):
                intersection.append(tc1[i])

    return intersection
            
def calculate_novelty(tc1, tc2):
    """
    > The novelty of two test cases is the proportion of states that are unique to each test case
    We implement novelty calculation according to Jaccard distance definition:
    intersection/(set1 size + set2 size - intersection)
    
    :param tc1: The first test case
    :param tc2: The test case that is being compared to the test suite
    :return: The novelty of the two test cases.
    """
    intersection = intersect(tc1, tc2)
    total_states = len(tc1) + len(tc2) - len(intersection)

    novelty = 1 - len(intersection) / total_states
    return -novelty