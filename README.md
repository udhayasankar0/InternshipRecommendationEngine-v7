### **`README.md`**

```markdown
# Lightweight Hybrid Internship Recommender

This project provides a lightweight hybrid internship recommender system. It takes a user's profile (skills, qualifications, location preferences, etc.) and recommends the top K most suitable internships from a given dataset.

The recommender uses a multi-stage process to ensure relevant and personalized recommendations.

## How It Works

1.  **Hard Skill Filter**: The system first applies a hard filter to discard any internships that do not match at least one of the user's core skills.
2.  **Component Scoring**: Each remaining internship is then scored across several dimensions:
    *   **Semantic Similarity (SBERT)**: A pre-trained SBERT model (`all-MiniLM-L6-v2`) calculates the contextual match between the user's profile and the internship description.
    *   **Skill Overlap**: A ratio is calculated based on how many of the user's skills are present in the internship requirements.
    *   **Location**: Internships in the user's preferred location are scored higher. Remote options are also considered if the user is open to them.
    *   **Stipend**: A continuous score is given based on how well the internship's stipend meets the user's minimum expectation.
    *   **Application Deadline**: A check is performed to ensure the application deadline has not passed relative to the user's availability.
3.  **Final Weighted Score**: The individual scores are combined using a weighted average to produce a final relevance score for each internship.
4.  **Top K Recommendations**: The internships are ranked by their final score, and the top K results are returned.

## Features

*   **Hybrid Approach**: Combines hard filtering with a weighted scoring of multiple signals for accuracy.
*   **Semantic Matching**: Uses a pre-trained SBERT model for nuanced text understanding.
*   **Explainability**: Provides "why this fits" tags for each recommendation, making the results transparent to the user.
*   **Caching**: Caches job embeddings to significantly speed up subsequent runs.
*   **Configurable**: Key parameters like scoring weights, model name, and file paths are centralized in a `config.py` file for easy modification.

## Project Structure

v_5/
├── .gitignore
├── config.py
├── dataset/
│ └── BANGALORE.csv
├── main.py
├── model_utils.py
├── rules.py
├── geolocation.py
├── explainers.py
├── requirements.txt
├── README.md
└── user.json



## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd v_5
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Customize the user profile** by editing `user.json` with your skills, preferred location, and other preferences.
2.  **Run the recommender** from the command line:
    ```bash
    python main.py
    ```
3.  The system will output a JSON object containing the top 5 recommended internships.

You can also customize the run with command-line arguments:
```bash
python main.py --dataset /path/to/your/data.csv --user /path/to/your/profile.json --k 10