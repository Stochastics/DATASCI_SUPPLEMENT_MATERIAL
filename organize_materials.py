import os
import shutil

# Configuration: { "Original Name": ("New Name", "Folder", "Student Description") }
curriculum_data = {
    # Phase 1: Python Basics
    "aboutloops.md": ("Python_Basics_For_vs_While_Loops.md", "Phase_01_Python_Basics", "Deep dive into 'for' vs 'while' loops with Data Science use cases."),
    "PythonIntro.md": ("Python_Basics_Data_Structures_Reference.md", "Phase_01_Python_Basics", "Reference guide for Lists, Tuples, Sets, and Dictionaries."),
    
    # Phase 2: Intro to Data Science
    "edge_cases.md": ("DS_Data_Cleaning_Edge_Cases_Cheat_Sheet.md", "Phase_02_DS_Intro", "Essential guide for handling messy real-world data like nulls and malformed text."),
    "Example TOY ML Example Customer Sales prediction .pdf": ("DS_Intro_ML_Workflow_Pipeline.pdf", "Phase_02_DS_Intro", "Visual workflow of a standard Machine Learning pipeline."),
    
    # Essentials: SQL
    "sql_cheat_sheet.md": ("SQL_Basics_Joins_CTEs_Stored_Procedures.md", "Essentials_03_SQL", "Comprehensive SQL reference from basic SELECTs to advanced recursive CTEs."),
    
    # Foundations I: Cloud
    "image.png": ("Cloud_Computing_Spark_Architecture.png", "Foundations_I_04_Cloud", "Diagram showing Spark Master/Worker node architecture."),
    "spark-components.png": ("Cloud_Spark_Cluster_Components_Diagram.png", "Foundations_I_04_Cloud", "Overview of Spark Driver, Cluster Manager, and Executor relationship."),
    "simple_git_cmd.md": ("Cloud_Git_Workflow_Essentials.md", "Foundations_I_04_Cloud", "Quick-start guide for everyday Git commands and remote workflows."),
    "mock_pyspark_emr_project.md": ("Cloud_Mock_Project_PySpark_EMR_Airflow.md", "Foundations_I_04_Cloud", "Architectural blueprint for orchestrating PySpark jobs with Airflow and EMR."),
    "pyspark_adtech_example.py": ("Cloud_PySpark_AdTech_ETL_Example.py", "Foundations_I_04_Cloud", "Real-world AdTech ETL code demonstrating broadcast joins and UDFs."),
    
    # Foundations I: Stats
    "basic_probability_cheat_sheet.pdf": ("Stats_Probability_Fundamentals_Cheat_Sheet.pdf", "Foundations_I_05_Stats", "Core math rules for probability, including Bayes' Theorem."),
    "statistical_tests.pdf": ("Stats_Hypothesis_Testing_Cheat_Sheet.pdf", "Foundations_I_05_Stats", "Decision matrix for choosing between Parametric and Non-Parametric tests."),
    
    # Foundations I: Regression
    "Linear_Regression_Cheat_Sheet.pdf": ("Regression_Linear_Multiple_Regularization.pdf", "Foundations_I_06_Regression", "Mathematical foundations of Linear Regression and Regularization (Lasso/Ridge)."),
    "logistic_regression_cheat_sheet.pdf": ("Regression_Logistic_Classification_Metrics.pdf", "Foundations_I_06_Regression", "Guide to Logistic Regression and classification performance metrics (ROC/AUC)."),
}

def build_ds_supplements(base_path):
    folder_contents = {}

    for original_name, (new_name, folder, description) in curriculum_data.items():
        old_file_path = os.path.join(base_path, original_name)
        
        if os.path.exists(old_file_path):
            target_folder = os.path.join(base_path, folder)
            os.makedirs(target_folder, exist_ok=True)
            
            # Track which folder gets which file for the README
            if folder not in folder_contents:
                folder_contents[folder] = []
            folder_contents[folder].append((new_name, description))
            
            # Move and rename
            shutil.move(old_file_path, os.path.join(target_folder, new_name))
            print(f"✅ Organized: {new_name} -> DS_SUPPLEMENTS/{folder}")

    # Generate README.md in each folder
    for folder, files in folder_contents.items():
        readme_path = os.path.join(base_path, folder, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"# 📘 DS_SUPPLEMENTS: {folder.replace('_', ' ')}\n\n")
            f.write("> **Notice:** These are independent supplemental notes created by the facilitator. They are intended to bridge the gap between curriculum and industry practice and are not official school documentation.\n\n")
            f.write("## Phase Resources\n")
            f.write("| File Name | Description |\n|---|---|\n")
            for name, desc in files:
                f.write(f"| `{name}` | {desc} |\n")
            f.write("\n---\n*Supplemental Materials for Cohort Success*")
        print(f"📝 Generated README.md for {folder}")

if __name__ == "__main__":
    # Ensure this runs in the folder containing your messy files
    build_ds_supplements(".")