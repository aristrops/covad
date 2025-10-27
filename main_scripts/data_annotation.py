import json
import os
import pandas as pd
import argparse

from utils import dataset_utils


def create_concept_dataset(dataset: str,
                           dataset_path: str,
                           category: str):
    
    # #step 1: create concept list
    # print(f"\nCreating concept dataset for {category} category...")
    # print("\nQuerying the VLM to extract the concept list...")
    # concept_list = dataset_utils.create_concept_list(dataset, dataset_path, category)

    # #step 2: aggregate and filter concept list
    # print("\nReducing the dimensionality of the concept set...")
    # filtered_concept_list = dataset_utils.aggregate_concepts(concept_list, category)
    # print("Removing too similar concepts...")
    # filtered_concept_list = dataset_utils.compute_concept_similarity(filtered_concept_list)
    # final_concepts = dataset_utils.compute_class_similarity(filtered_concept_list, category)
    # print(f"Final number of concepts kept for {category} category: {len(final_concepts)}")

    # with open(f"concept_lists/filtered/{dataset}/{category}_concepts.json", "w") as f:
    #     json.dump(final_concepts, f)
    
    # #step 3: create final dataset
    # print("\nAutomatically annotating the dataset...")
    # final_df = dataset_utils.create_final_dataset(dataset_path, dataset, category, final_concepts)

    final_df = pd.read_csv(f"/mnt/disk1/arianna_stropeni/cbm_data/{dataset}/{category}_dataset_automated.csv")
    with open(f"concept_lists/filtered/{dataset}/{category}_concepts.json", "r") as f:
        final_concepts = json.load(f)

    print(final_df.head())                

    #step 4: split the dataset into train, test and validation
    print("\nSplitting the dataset into train, test and validation...")
    final_df = dataset_utils.modify_columns(final_df)
    print(final_df.head())
    final_df = dataset_utils.split_dataframe(final_df)
    print(final_df.head())

    #step 5: remove uninformative and highly correlated concepts
    print("\nRemoving uninformative and highly correlated concepts...")
    final_df, remaining_concepts = dataset_utils.drop_concepts(final_df, final_concepts)
    print(final_df.head())
    final_df, remaining_concepts = dataset_utils.compute_correlation(final_df, remaining_concepts)
    print(final_df.head())

    save_path = f"/mnt/disk1/arianna_stropeni/cbm_data/{dataset}/{category}_dataset_automated.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    final_df.to_csv(save_path, index = False)
    print(f"Final dataframe saved in {save_path}")

    # #step 6: statistical analysis of concepts
    # print("\nComputing statistical significance of concepts...")
    # dataset_utils.chi_square_test(final_df, remaining_concepts)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="MVTec or RealIAD dataset")
    parser.add_argument("--categories", type = str, nargs="+", help="Which categories to annotate")

    args = parser.parse_args()

    #paths to datasets
    if args.dataset == "mvtec":
        dataset_path = "/mnt/disk1/borsattifr/datasets/mvtec"
    elif args.dataset == "realiad":
        dataset_path = "/mnt/disk1/yfbenkhalifa/datasets/realiad/realiad_256"

    for category in args.categories:
        create_concept_dataset(args.dataset, dataset_path, category)

if __name__ == "__main__":
    main()
    