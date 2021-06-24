# END2.0_Assignment07



# Part 1

## Index
- Dataset used: datasetSentences.txt. (no augmentation required)
- Dataset is having around 11-12k examples.
- Dataset has been splitted into 70/30 Train and Test (no validation set has been used)
- Floating-point labels have been converted into 5 classes (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0) 
- Code for Dataset preparation.
- training log text 
- prediction on 10 samples picked from the test dataset

 ## Dataset
 We have downloaded the StanfordSentimentAnalysis Dataset  from the given link. We used "datasetSentences.txt" and "sentiment_labels.txt" files from the zip folder. This dataset contains just over 11,000 pieces of Stanford data from HTML files of Rotten Tomatoes. The sentiments are rated between 1 and 5, where one is the most negative and 5 is the most positive.
 
## Examples 

![Examples](Assets/part1_dataset_examples.PNG)

## Splitting

train_data,test_data = train_test_split(df,test_size=0.3)
train_data.reset_index(inplace=True)
test_data.reset_index(inplace=True)

## Floating-point labels have been converted into 5 classes (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0) 

def create_label(label):
    if label <= 0.2: return 0
    if label <= 0.4: return 1
    if label <= 0.6: return 2
    if label <= 0.8: return 3
    return 4

## Sentiment label data
sentiment_labels = pd.read_csv("sentiment_labels.txt", names=['phrase_ids', 'labels'], sep="|", header=0)
sentiment_labels['labels'] = sentiment_labels['labels'].apply(create_label)
## sentence index and sentence
sentence_ids = pd.read_csv("datasetSentences.txt", sep="\t")
## phrases and phrase_ids
dic = pd.read_csv("dictionary.txt", sep="|", names=['phrase', 'phrase_ids'])
## Train Test split
train_test_valid_split = pd.read_csv("datasetSplit.txt")
## Merging data to create final data
sentence_phrase_merge = pd.merge(sentence_ids, dic, left_on='sentence', right_on='phrase')
sentence_phrase_split = pd.merge(sentence_phrase_merge, train_test_valid_split, on='sentence_index')
dataset = pd.merge(sentence_phrase_split, sentiment_labels, on='phrase_ids')
print("Length of data : ",dataset.shape[0])
## Subset data for Model
df = dataset[['labels','sentence']]
print('train data shape: ', df.shape)
df.head()

## Code for Dataset preparation

def cleanup_text(texts):
    cleaned_text = []
    for text in texts:
        # remove punctuation
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        # remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # remove newline
        text = re.sub(r'\n', ' ', text)
        text = str(text).lower()
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text) # remove URLs
        text = re.sub('@[^\s]+', 'ATUSER', text) # remove usernames
        text = re.sub(r'#([^\s]+)', r'\1', text) # remove the # in #hashtag
        text = re.sub('[^A-Za-z0-9]+', ' ', text) # remove # and numbers
        cleaned_text.append(text)
    return cleaned_text
    
  
 ## training log text  
 
 Train Loss: 1.577 | Train Acc: 27.86%
	 Val. Loss: 1.564 |  Val. Acc: 30.89% 

	Train Loss: 1.539 | Train Acc: 34.75%
	 Val. Loss: 1.555 |  Val. Acc: 32.17% 

	Train Loss: 1.497 | Train Acc: 39.25%
	 Val. Loss: 1.552 |  Val. Acc: 31.70% 

	Train Loss: 1.456 | Train Acc: 43.77%
	 Val. Loss: 1.560 |  Val. Acc: 31.10% 

	Train Loss: 1.412 | Train Acc: 48.70%
	 Val. Loss: 1.559 |  Val. Acc: 32.38% 

	Train Loss: 1.373 | Train Acc: 52.67%
	 Val. Loss: 1.561 |  Val. Acc: 32.41% 

	Train Loss: 1.330 | Train Acc: 57.11%
	 Val. Loss: 1.571 |  Val. Acc: 31.87% 

	Train Loss: 1.297 | Train Acc: 60.59%
	 Val. Loss: 1.567 |  Val. Acc: 32.56% 

	Train Loss: 1.271 | Train Acc: 63.27%
	 Val. Loss: 1.575 |  Val. Acc: 31.43% 

	Train Loss: 1.253 | Train Acc: 65.10%
	 Val. Loss: 1.587 |  Val. Acc: 30.30%
  


## Assignment details on quiz section:
Upload to github and proceed to answer these questions asked in the S7 - Assignment Solutions, where these questions are asked:
Share the link to your github repo (100 pts for code quality/file structure/model accuracy)
Share the link to your readme file (200 points for proper readme file)
Copy-paste the code related to your dataset preparation (100 pts)
Share your training log text (you MUST have been testing for test accuracy after every epoch) (200 pts)
Share the prediction on 10 samples picked from the test dataset. (100 pts)

# Part 2

Train model we wrote in the class on the following two datasets taken from this link (Links to an external site.): 
http://www.cs.cmu.edu/~ark/QA-data/ (Links to an external site.)
https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs (Links to an external site.)

## Assignment details on quiz section:
Once done, please upload the file to GitHub and proceed to answer these questions in the S7 - Assignment Solutions, where these questions are asked:
Share the link to your GitHub repo (100 pts for code quality/file structure/model accuracy) (100 pts)
Share the link to your readme file (100 points for proper readme file), this file can be the second part of your Part 1 Readme (basically you can have only 1 Readme, describing both assignments if you want) (100 pts)
Copy-paste the code related to your dataset preparation for both datasets.  (100 pts)
