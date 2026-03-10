from pathlib import Path


def create_structure(base_dir, new_dir_name, sub_folders_and_files):
    base_dir = Path(f"{base_dir}/{new_dir_name}")
    for module, sub_folders in sub_folders_and_files.items():
        module_path = base_dir / module
        module_path.mkdir(parents=True, exist_ok=True)
        for sub_folder, files in sub_folders.items():
            sub_folder_path = module_path / sub_folder
            sub_folder_path.mkdir(parents=True, exist_ok=True)
            for file in files:
                file_path = sub_folder_path / file
                file_path.touch()


def main():
    print("Hello from helpers-py!")


if __name__ == "__main__":
    create_structure(
        base_dir="/Users/siddhantangore/Code/GitHub/ai_ml_study/Courses/IBM",
        new_dir_name="Course_2_Build_RAG_Applications_Get_Started",
        sub_folders_and_files={
            "Module_1": {
                "Welcome_to_the_Course": [
                    "Course_Introduction.md",
                    "Course_Overview.md",
                    "RAG_&_Agentic_AI_Professional_Certificate_Overview.md",
                    "Helpful_tips_for_course_completion.md",
                ],
                "What_is_RAG?": [
                    "Why_RAG?.md",
                    "More_RAG_Details.md",
                    "What_is_RAG?.md",
                    "Summarize_Private_Docs_using_RAG_LangChain_&_LLms.md",
                    "Quiz_What_is_RAG?.md",
                ],
                "Summary_&_Evaluation": [
                    "Summary_Intro_to_RAG.md",
                    "Cheat_Sheet_Intro_to_RAG.md",
                    "Quiz_Intro_to_RAG.md",
                ],
            },
            "Module_2": {
                "Create_an_Interactive_RAG_Application_with_User_Friendly_Gradio_Interface": [
                    "Getting_started_with_Gradio.md",
                    "Intro_to_Gradion.md",
                    "Setup_simple_Gradio_Interface_to_Interact_with_Your_Models.md",
                    "Construct_a_QA_BOT_with_LangChain_&_LLM_to_Answer_Questions_from_Loaded_Docs.md",
                    "Quiz_Building_Apps_with_RAG.md",
                ],
                "Summary_&_Evaluation": [
                    "Summary_Building_Apps_with_RAG.md",
                    "Cheat_Sheet_Building_Apps_with_RAG.md",
                    "Quiz_Building_Apps_with_RAG.md",
                ],
            },
            "Module_3": {
                "Application_Dev_with_LlamaIndex": [
                    "Intro_to_LlamaIndex_Doc_Ingestion_&_Chunking.md",
                    "Intro_to_LlamaIndex_From_VectorDB_to_Query_Engines.md",
                    "LangChain_&_LlamaIndex_compared.md",
                    "Build_an_AI_Icrebreak_Bot_with_IBM_Granite_LlamaIndex.md",
                    "Quiz_Application_Dev_with_LlamaIndex.md",
                ],
                "Summary_&_Evaluation": [
                    "Summary_Build_RAG_Apps_with_LlamaIndex.md",
                    "Cheat_Sheet_Build_RAG_Apps_with_LlamaIndex.md",
                    "Quiz_Build_RAG_Apps_with_LlamaIndex.md",
                ],
                "Course_Wrap_up": [
                    "Course_Wrap_up.md",
                    "Congratulations_&_Next_Steps.md",
                    "Team_&_Acknowledgements.md",
                ],
            },
        },
    )
