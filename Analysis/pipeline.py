# main pipeline for exploring the loglike data
# in this file, we:
# 1. find the maximum log-likelihood value
# 2. find the structures with the maximum log-likelihood value
# 3. find the average count of the structures with the maximum log-likelihood value
# 4. find the structures with count >= average count
# 5. find the average jaccard similarity of the structures with count >= average count
# 6. find the top 10 log-likelihood values and their iteration
# 7. find the average jaccard similarity of the top 10 log-likelihood values
# 8. find the average cosine similarity of the top 10 log-likelihood values
# 9. export the data to JSON files

import os
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

def jaccard_similarity_line(line1, line2):
    set1 = set(line1.split())
    set2 = set(line2.split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union != 0 else 0


networks = [
    "Davis/Davis_southern_club_women_two_mode",
    "clique/clique_cp",
    "football/football",
    "OF/OF_long_bin_Gv1",
    "pollbooks/polbooks",
    "surfers/surfers",
    "Twitter1/twitter_1",
    "polblogs/polblogs",
    "airlines/airlines",
    "as-22july06/as-22july06",
    "FB circles/facebookCircles_G",
    "FB pages/facebookPages_G",
    "Madrid/madridTerror_G",
    "1997/as08nov1997_G"
    
]

for network_path in networks:
    # network, type = network_path.split("/")
    file_path = f"data/{network_path}_ll.txt"  # loglike data
    log_likelihood_values = np.loadtxt(file_path)
    
    with open(f"data/{network_path}_configs.txt", "r") as file:  # configs
        lines = file.readlines()

    with open(f"data/{network_path}_pairs.txt", "r") as pfile:  # pairs
        pairs_lines = pfile.readlines()
        
    network, type = network_path.split("/")
        
    max_loglike = np.max(log_likelihood_values)
    print(f"Maximum Log-Likelihood for {network}: {max_loglike}")
    
    structures_dict = {}

    for i in range(len(log_likelihood_values)):
        if log_likelihood_values[i] == max_loglike:
            structure = lines[i].strip()
            pairs = pairs_lines[i].strip()  # Assuming pairs_lines contains values from clique_cp_pairs.txt
            structures_dict[structure] = pairs

    for i, (structure, pairs) in enumerate(structures_dict.items()):
        print(f"Index {i}: Structure: {structure}, Pairs: {pairs}")
        
    unique_structures = set()  # set to get unique structures

    for i in range(len(log_likelihood_values)):
        if log_likelihood_values[i] == max_loglike:
            structure = lines[i].strip()  # Remove leading/trailing whitespaces
            unique_structures.add(structure)

    unique_structures_list = list(unique_structures)

    for i, unique_structure in enumerate(unique_structures_list):
        print(f"Index {i}: {unique_structure}")    
        
    structures_info = {}  # Dictionary to store structure, count, and indexes

    for i in range(len(log_likelihood_values)):
        if log_likelihood_values[i] == max_loglike:
            structure = lines[i].strip()  # Remove leading/trailing whitespaces

            if structure in structures_info:
                # Increment count
                structures_info[structure]['count'] += 1
                # Append index
                structures_info[structure]['indexes'].append(i)
            else:
                # Add new entry
                structures_info[structure] = {'count': 1, 'indexes': [i]}

    for i, (structure, info) in enumerate(structures_info.items()):
        print(f"Index {i}: Structure: {structure}, Count: {info['count']}, Indexes: {info['indexes']}")
        
    # take average of counts in structures_info
    counts = [info['count'] for info in structures_info.values()]
    avg_count = np.mean(counts)
    print(f"Average count for {network}: {avg_count}")

    # keep structures with count >= avg_count in a new dictionary
    structures_info_filtered = {}

    for structure, info in structures_info.items():
        if info['count'] >= avg_count:
            structures_info_filtered[structure] = info
            
    for i, (filtered_structure, filtered_info) in enumerate(structures_info_filtered.items()):
        print(f"Index {i}: Structure: {filtered_structure}, Count: {filtered_info['count']}, Indexes: {filtered_info['indexes']}")
        
    # create a list of structures again, so we can get binary values
    temp_lines = []
    for i, (filtered_structure, filtered_info) in enumerate(structures_info_filtered.items()):
        temp_lines.append(filtered_structure)
        
    binary_lines = {}

    # convert to binary, store in dictionary
    for i in range(len(temp_lines)):
        temp = ''
        for j in temp_lines[i].split():
            temp += f'{bin(int(j))[2:]} '
        binary_lines[i] = temp
        
    total_similarity = 0
    pair_count = 0

    # test jaccard sim for the filtered structures
    for i in range(len(temp_lines[-5:])):
        for j in range(i + 1, len(temp_lines[-5:])):
            # JS for each pair of lines
            similarity = jaccard_similarity_line(binary_lines[i], binary_lines[j])
            
            # sum similarity 
            total_similarity += similarity
            pair_count += 1

    # average similarity
    average_similarity = total_similarity / pair_count if pair_count != 0 else 0

    print(f"Average JS for {network}: {average_similarity}")
    
    # ----------------- top 10 config -----------------
    
    # show top 10 log-likelihood values and their iteration
    top_10_indices = np.argsort(log_likelihood_values)[-10:]
    print(f"Top 10 log-likelihood values and their iteration for {network}:")
    print("iteration\tloglike")
    for i in top_10_indices:
        print(i, "\t\t", log_likelihood_values[i])
        
    # get the configs for the top 10 log-likelihood values
    top_10_configs = []
    for i in top_10_indices:
        top_10_configs.append(lines[i].strip())
    print(f"Top 10 configs for {network}:")
    for i in top_10_configs:
        print(i)

    # convert to binary, store in dictionary
    binary_lines = {}

    for i in range(len(top_10_configs)):
        temp = ''
        for j in top_10_configs[i].split():
            temp += f'{bin(int(j))[2:]} '
        binary_lines[i] = temp
        
    total_similarity = 0
    pair_count = 0

    # test jaccard sim for the top 10 configs
    for i in range(len(top_10_configs)):
        for j in range(i + 1, len(top_10_configs)):
            # JS for each pair of lines
            similarity = jaccard_similarity_line(binary_lines[i], binary_lines[j])
            
            # sum similarity 
            total_similarity += similarity
            pair_count += 1
            
    

    
    # average similarity
    average_similarity_top_10  = total_similarity / pair_count if pair_count != 0 else 0

    print(f"Average JS for Top 10 {network}: {average_similarity_top_10}")  
    
    # put all binary_lines into binary_matrix
    binary_matrix_top_10 = np.array([[int(bit) for bit in line.split()] for line in binary_lines.values()])
    cosine_sim_top_10 = cosine_similarity(binary_matrix_top_10)

    average_csimilarity_top_10 = np.mean(cosine_sim_top_10)

    print(f"Average Cosine Similarity for Top 10 {network}: {average_csimilarity_top_10}")


    # export to JSON
    json_filename = f"output_{network}.json"
    
    # print what we're exporting
    print(f"network: {network}")
    print(f"max_loglike: {int(max_loglike)}")
    print(f"structures: {structures_dict}")
    print(f"average_similarity: {average_similarity}")
    print(f"structures_info_filtered: {structures_info_filtered}")
    print(f"top_10_log_likelihood: {top_10_indices}")
    print(f"top_10_configs: {top_10_configs}")
    print(f"average_similarity_top_10: {average_similarity_top_10}")

    data_to_export = {
        "network": network,
        "max_loglike": int(max_loglike),
        "structures": [{"index": i, "structure": structure, "pairs": pairs} for i, (structure, pairs) in enumerate(structures_dict.items())],
        "average_similarity": average_similarity,
        "structures_info_filtered": [{"index": i, "structure": filtered_structure, "count": filtered_info['count'], "indexes": filtered_info['indexes']} for i, (filtered_structure, filtered_info) in enumerate(structures_info_filtered.items())],
        "top_10_log_likelihood": [{"iteration": int(i), "loglike": float(log_likelihood_values[i])} for i in top_10_indices],
        "top_10_configs": top_10_configs,
        "average_similarity_top_10": average_similarity_top_10,
        "cosine_similarity_top_10": average_csimilarity_top_10,  
    }

    with open(json_filename, 'w') as json_file:
        json.dump(data_to_export, json_file, indent=4)

    print(f"Data exported to {json_filename}")
