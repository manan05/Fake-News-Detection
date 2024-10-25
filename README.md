# Setting up
To create and activate the virtual environment for this repository, follow these steps:

1. Create the virtual environment:
    ```sh
    python -m venv env
    ```

2. Activate the virtual environment:
    - On Windows:
        ```sh
        .\env\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source env/bin/activate
        ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```


# About the Dataset 

### Columns Overview:
1. **`id`**: A unique identifier for each statement. This is typically used for reference and has no direct impact on the classification task.

2. **`label`**: The target variable representing the truthfulness of the statement. In the original dataset, this column likely contains categorical values indicating different levels of truth, such as:
   - `0`: Pants on Fire (completely false)
   - `1`: False
   - `2`: Mostly False
   - `3`: Half True
   - `4`: Mostly True
   - `5`: True
   
   In this script, these labels have been mapped to a binary classification:
   - `0-2` (Pants on Fire, False, Mostly False) are mapped to `0` (Fake).
   - `3-5` (Half True, Mostly True, True) are mapped to `1` (Not Fake).

3. **`statement`**: The actual text of the statement or claim being evaluated. This is the primary feature used for prediction, and it undergoes preprocessing steps like tokenization and stemming before being fed into the model.

4. **`subject`**: The topic or category of the statement (e.g., taxes, healthcare, etc.). This column can provide additional context but isn't used in the current model.

5. **`speaker`**: The person or organization making the statement. This could be used to add features based on the speaker's historical truthfulness but isn't used in this script.

6. **`speaker_description`**: Additional details about the speaker, such as their profession or role.

7. **`state_info`**: Information about the location or state where the statement was made. It could be used for feature engineering.

8. **Fact-Checking Columns (`true_counts`, `mostly_true_counts`, `half_true_counts`, etc.)**: 
   - These columns provide metadata on how many times statements by the same speaker have been categorized into different truth levels. For example, how many times the speaker has been rated as "True" or "False".
   - These columns could potentially be used to add credibility-related features for the model.

9. **`context` and `justification`**: These columns might provide additional textual information about the context in which the statement was made and the justification for its truthfulness rating.

### Purpose of the Dataset:
The goal is to predict whether a given statement is fake or not based on the `statement` text. The dataset contains a mix of statements that have been fact-checked and classified into different truth categories. By converting these categories into a binary classification (fake or not fake), the model aims to detect potentially deceptive or misleading information.

### Typical Use Cases:
1. **Fake News Detection**: Automatically flagging misleading or false information in online content.
2. **Fact-Checking Automation**: Assisting human fact-checkers by prioritizing which claims may need verification.
3. **Media Literacy Tools**: Providing users with indicators about the reliability of information sources.
WW