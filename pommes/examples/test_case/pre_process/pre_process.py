from pommes.utils import split_excel_to_csv


def pre_process_test_case(list_excel_files, repo_data):
    for excel_file in list_excel_files:
        split_excel_to_csv(input_excel_path=excel_file, output_folder=repo_data)


if __name__ == "__main__":
    repo = "study/test_case/data"
    excel_file1 = "study/test_case/raw_data/excel_1.xlsx"
    excel_file2 = "study/test_case/raw_data/excel_2.xlsx"
    excel_files = [excel_file1, excel_file2]

    pre_process_test_case(excel_files, repo)
