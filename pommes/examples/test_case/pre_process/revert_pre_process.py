from pommes.utils import combine_csv_to_excel

if __name__ == "__main__":
    repo = "study/test_case/data"
    excel_file = "study/test_case/raw_data/excel_1.xlsx"

    combine_csv_to_excel(repo, excel_file)
