# Sentiment Analysis in Python

### Link to the dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
## Overview
This project performs sentiment analysis on the **Amazon Fine Food Reviews** dataset using two different techniques:
1. **VADER (Valence Aware Dictionary and Sentiment Reasoner)** - A lexicon and rule-based sentiment analysis tool.
2. **Roberta Pretrained Model from Hugging Face** - A transformer-based approach.

## Dataset
The dataset is sourced from Kaggle and contains user reviews of Amazon products. It is downloaded using `kagglehub`.

### Dataset Structure
- `Id`: Unique identifier for the review.
- `ProductId`: Unique identifier for the product.
- `UserId`: Unique identifier for the user.
- `ProfileName`: Name of the reviewer.
- `HelpfulnessNumerator` and `HelpfulnessDenominator`: Measures of review helpfulness.
- `Score`: Star rating given by the user.
- `Time`: Timestamp of the review.
- `Summary`: Short summary of the review.
- `Text`: The review text.

## Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK (Natural Language Toolkit)
- Transformers (Hugging Face)
- SciPy
- TQDM

## Project Steps
### 1. Data Loading and Exploration
- Load the dataset and select the first 500 rows for analysis.
- Perform exploratory data analysis (EDA) by visualizing the distribution of review scores.

### 2. VADER Sentiment Analysis
- Use `SentimentIntensityAnalyzer` from NLTK to compute sentiment scores (positive, neutral, negative, and compound) for each review.
- Visualize sentiment scores across different star ratings.

### 3. Roberta Pretrained Model
- Use the `cardiffnlp/twitter-roberta-base-sentiment` model from Hugging Face.
- Tokenize and run the model to get sentiment scores.

### 4. Combining Results
- Merge the VADER and Roberta sentiment scores into a single dataframe.
- Compare the results and analyze discrepancies between the models.

### 5. Sentiment Analysis with Transformers Pipeline
- Use the `pipeline` function from Hugging Face for quick and easy sentiment predictions.

## Results
- Visual comparisons of sentiment scores between the models.
- Identification of positive 1-star and negative 5-star reviews.

## Installation
1. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk transformers kagglehub tqdm scipy twython
   ```
2. Download the dataset from Kaggle.
3. Run the Jupyter Notebook or Colab notebook provided.

## Running the Code
```python
!pip install kagglehub
import kagglehub

# Download the dataset
organizations_snap_amazon_fine_food_reviews_path = kagglehub.dataset_download('organizations/snap/amazon-fine-food-reviews')
```

## Visualizations
- Bar plots showing sentiment scores across different review ratings.
- Pair plots to compare VADER and Roberta sentiment scores.

## Conclusion
- VADER is effective for basic sentiment analysis but can be limited by contextual understanding.
- Roberta, being a transformer model, captures deeper context and provides more accurate sentiment scores.

## Future Work
- Fine-tuning Roberta on domain-specific data.
- Incorporating other sentiment analysis models for comparison.

## References
- [Amazon Fine Food Reviews Kaggle Dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- [Hugging Face Transformers Library](https://huggingface.co/transformers)
- [NLTK Documentation](https://www.nltk.org/)

## Contact
For any questions or suggestions, feel free to reach out to **Nanda Kishore Reddy Gajjala Venkata**.

