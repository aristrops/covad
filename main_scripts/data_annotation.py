import json
import os
import argparse

from utils import dataset_utils

def create_concept_dataset(dataset_path: str,
                           category: str,
                           model_name: str,
                           save_path: str, 
                           use_gen_anomalies: bool = False):
    
    #step 1: create concept list
    print(f"\nCreating concept dataset for {category} category...")
    print("\nQuerying the VLM to extract the concept list...")
    concept_list = dataset_utils.create_concept_list(dataset_path, category, model_name, use_gen_anomalies=use_gen_anomalies)

    #step 2: aggregate and filter concept list
    print("\nReducing the dimensionality of the concept set...")
    filtered_concept_list = dataset_utils.aggregate_concepts(concept_list, category, model_name)
    print("Removing too similar concepts...")
    filtered_concept_list = dataset_utils.compute_concept_similarity(filtered_concept_list)
    final_concepts = dataset_utils.compute_class_similarity(filtered_concept_list, category)
    print(f"Final number of concepts kept for {category} category: {len(final_concepts)}")

    if use_gen_anomalies:
        output_dir_filtered = f"concept_lists/filtered//gen_anomalies"
    else:
        output_dir_filtered = f"concept_lists/filtered"
    os.makedirs(output_dir_filtered, exist_ok=True)

    with open(os.path.join(output_dir_filtered, f"{category}_concepts.json"), "w") as f:
        json.dump(final_concepts, f)
    
    #step 3: create final dataset
    print("\nAutomatically annotating the dataset...")
    final_df = dataset_utils.create_final_dataset(dataset_path, category, final_concepts, model_name, use_gen_anomalies = use_gen_anomalies)        

    #step 4: split the dataset into train, test and validation
    print("\nSplitting the dataset into train, test and validation...")
    final_df = dataset_utils.modify_columns(final_df)
    final_df = dataset_utils.split_dataframe(final_df)

    #step 5: remove uninformative and highly correlated concepts
    print("\nRemoving uninformative and highly correlated concepts...")
    final_df, remaining_concepts = dataset_utils.drop_concepts(final_df, final_concepts)
    final_df, remaining_concepts = dataset_utils.compute_correlation(final_df, remaining_concepts)

    print(f"Final number of concepts kept for {category} category after dataset annotation: {len(remaining_concepts)}")

    final_df.to_csv(save_path, index = False)
    print(f"Final dataframe saved in {save_path}")

    #step 6: statistical analysis of concepts
    print("\nComputing statistical significance of concepts...")
    dataset_utils.chi_square_test(final_df, remaining_concepts)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--categories", type = str, nargs="+", help="Which categories to annotate")
    parser.add_argument("--model_name", type=str, help="Which VLM to use")
    parser.add_argument("--save_path", type=str, help="Save path for concept dataset")
    parser.add_argument("--use_gen_anomalies", action = "store_true", help = "Annotate dataset with the generated anomalies")

    args = parser.parse_args()

    for category in args.categories:
        create_concept_dataset(args.dataset_path, category, args.model_name, args.save_path, args.use_gen_anomalies)

if __name__ == "__main__":
    main()
    