# Conversation Emotion Model

This project uses dialogue data from the TV show "Friends" to train and evaluate models for emotion classification in conversational text.

## Project Structure

- [`emorynlp_train_final.csv`](d:/LLM/conversation_emotion_model/emorynlp_train_final.csv): Main dataset containing utterances, speakers, annotated emotions, and metadata.
- [`config.json`](d:/LLM/conversation_emotion_model/config.json): Model configuration for BERT-based sequence classification.
- `label_encoder.pkl`: Serialized label encoder for emotion labels.
- `special_tokens_map.json`, `tokenizer_config.json`, `vocab.txt`: Tokenizer files for BERT.
- [`LLm_2.ipynb`](d:/LLM/conversation_emotion_model/LLm_2.ipynb): Jupyter notebook for data processing, model training, and evaluation.

## Dataset

Each row in [`emorynlp_train_final.csv`](d:/LLM/conversation_emotion_model/emorynlp_train_final.csv) contains:
- Utterance: The spoken line.
- Speaker: Character(s) who spoke.
- Emotion: Annotated emotion label (e.g., Joyful, Sad, Mad, Scared, Neutral, Powerful, Peaceful, Sad).
- Scene/Episode/Season/Timing: Metadata for context.

## Model

- BERT-based sequence classification (see [`config.json`](d:/LLM/conversation_emotion_model/config.json)).
- Multi-label emotion prediction.

## Usage

1. Install dependencies:
    ```sh
    pip install transformers scikit-learn pandas torch
    ```
2. Open [`LLm_2.ipynb`](d:/LLM/conversation_emotion_model/LLm_2.ipynb) in Jupyter or VS Code.
3. Run notebook cells to preprocess data, train, and evaluate the model.

## Outputs

- Classification metrics (hamming loss, classification report).
- Example predictions for sample utterances.

## Citation

If you use this dataset or code, please cite the original EmoryNLP dataset and HuggingFace Transformers.

## License

This project is for research and educational