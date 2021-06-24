# END2.0_Assignment07



# Part 1

## Index- Part 1
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

	Length of data :  11286
	train data shape:  (11286, 2)
	labels	sentence
	0	3	The Rock is destined to be the 21st Century 's...
	1	4	The gorgeously elaborate continuation of `` Th...
	2	2	Effective but too-tepid biopic
	3	3	If you sometimes like to go to the movies to h...
	4	4	Emerges as something rare , an issue movie tha...

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
	![image](https://user-images.githubusercontent.com/16242779/123306938-840c8100-d53f-11eb-8e2f-7a523fc0f30f.png)

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

## Hypperparameters
	lr = 1e-4
	batch_size = 16
	embedding_dim = 300
	dropout_keep_prob = 0.5
	seed = 42
	output_dim = 5
	hidden_dim1 = 512
	hidden_dim2 = 256
	n_layers = 2  # LSTM layers
	bidirectional = False 

## Model Architecture

	import torch.nn as nn
	import torch.nn.functional as F

	class LSTM(nn.Module):
	    def __init__(self, vocab_size, embedding_dim, hidden_dim1, hidden_dim2, output_dim, n_layers,
			 bidirectional, dropout):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.encoder = nn.LSTM(embedding_dim,
				    hidden_dim1,
				    num_layers=n_layers,
				    bidirectional=bidirectional,
				    batch_first=True)
		self.fc1 = nn.Linear(hidden_dim1*2 , hidden_dim2)
		self.fc2 = nn.Linear(hidden_dim2, output_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        
        packed_output, (hidden, cell) = self.encoder(packed_embedded)

        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        rel = self.relu(cat)
        dense1 = self.fc1(cat)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        # Final activation function softmax
        output = F.softmax(preds, dim=1)
            
        return output
  
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
## Output

- training loss vs validation Accuracy
	![image](https://user-images.githubusercontent.com/16242779/123307000-98e91480-d53f-11eb-9d3b-9ca88f916c98.png)


- training loss vs validation loss 
	![image](https://user-images.githubusercontent.com/16242779/123307332-0006c900-d540-11eb-8b9f-3ed973a6da4d.png)




- Confusion matrix-train data, predicted results
	![image](https://user-images.githubusercontent.com/16242779/123309335-6ab90400-d542-11eb-89d7-027e9af0213e.png)


- Confusion matrix-test data, predicted results
	![image](https://user-images.githubusercontent.com/16242779/123309390-799fb680-d542-11eb-86f4-cea591098870.png)




## prediction on 10 samples picked from the test dataset



****************************************
***** Correctly Classified Text: *******
****************************************
1) Text: Remember when Bond had more glamour than clamor ?
   Target Sentiment: Negative
   Predicted Sentiment: Negative

2) Text: While Broomfield 's film does n't capture the effect of these tragic deaths on hip-hop culture , it succeeds as a powerful look at a failure of our justice system .
   Target Sentiment: Positive
   Predicted Sentiment: Positive

3) Text: A giddy and provocative sexual romp that has something to say .
   Target Sentiment: Positive
   Predicted Sentiment: Positive

4) Text: The soul-searching deliberateness of the film , although leavened nicely with dry absurdist wit , eventually becomes too heavy for the plot .
   Target Sentiment: Negative
   Predicted Sentiment: Negative

5) Text: The movie attempts to mine laughs from a genre -- the gangster\/crime comedy -- that wore out its welcome with audiences several years ago , and its cutesy reliance on movie-specific cliches is n't exactly endearing .
   Target Sentiment: Negative
   Predicted Sentiment: Negative

6) Text: There 's something poignant about an artist of 90-plus years taking the effort to share his impressions of life and loss and time and art with us .
   Target Sentiment: Positive
   Predicted Sentiment: Positive

7) Text: But the movie 's narrative hook is way too muddled to be an effectively chilling guilty pleasure .
   Target Sentiment: Negative
   Predicted Sentiment: Negative

8) Text: The acting in Pauline And Paulette is good all round , but what really sets the film apart is Debrauwer 's refusal to push the easy emotional buttons .
   Target Sentiment: Very Positive
   Predicted Sentiment: Very Positive

9) Text: ... while certainly clever in spots , this too-long , spoofy update of Shakespeare 's Macbeth does n't sustain a high enough level of invention .
   Target Sentiment: Negative
   Predicted Sentiment: Negative

10) Text: In the book-on-tape market , the film of `` The Kid Stays in the Picture '' would be an abridged edition
   Target Sentiment: Negative
   Predicted Sentiment: Negative
   
   


****************************************
***** Incorrectly Classified Text: *******
****************************************
1) Text: The unique niche of self-critical , behind-the-scenes navel-gazing Kaufman has carved from Orleans ' story and his own infinite insecurity is a work of outstanding originality .
   Target Sentiment: Very Positive
   Predicted Sentiment: Positive

2) Text: Though Avary has done his best to make something out of Ellis ' nothing novel , in the end , his Rules is barely worth following .
   Target Sentiment: Negative
   Predicted Sentiment: Very Negative

3) Text: Successfully blended satire , high camp and yet another sexual taboo into a really funny movie .
   Target Sentiment: Positive
   Predicted Sentiment: Very Negative

4) Text: Plays like a checklist of everything Rob Reiner and his cast were sending up .
   Target Sentiment: Neutral
   Predicted Sentiment: Very Positive

5) Text: A sentimental hybrid that could benefit from the spice of specificity .
   Target Sentiment: Positive
   Predicted Sentiment: Very Positive

6) Text: When cowering and begging at the feet a scruffy Giannini , Madonna gives her best performance since Abel Ferrara had her beaten to a pulp in his Dangerous Game .
   Target Sentiment: Very Positive
   Predicted Sentiment: Neutral

7) Text: A very funny romantic comedy about two skittish New York middle-agers who stumble into a relationship and then struggle furiously with their fears and foibles .
   Target Sentiment: Positive
   Predicted Sentiment: Negative

8) Text: Once the expectation of laughter has been quashed by whatever obscenity is at hand , even the funniest idea is n't funny .
   Target Sentiment: Very Negative
   Predicted Sentiment: Very Positive

9) Text: Try as you might to scrutinize the ethics of Kaufman 's approach , somehow it all comes together to create a very compelling , sensitive , intelligent and almost cohesive piece of film entertainment .
   Target Sentiment: Very Positive
   Predicted Sentiment: Positive

10) Text: Boring and meandering .
   Target Sentiment: Very Negative
   Predicted Sentiment: Very Positive




## Assignment details on quiz section:
Upload to github and proceed to answer these questions asked in the S7 - Assignment Solutions, where these questions are asked:
Share the link to your github repo (100 pts for code quality/file structure/model accuracy)
Share the link to your readme file (200 points for proper readme file)
Copy-paste the code related to your dataset preparation (100 pts)
Share your training log text (you MUST have been testing for test accuracy after every epoch) (200 pts)
Share the prediction on 10 samples picked from the test dataset. (100 pts)

# Part 2

## Index- Part 2
- Dataset used:We used quora dataset from following resource location http://www.cs.cmu.edu/~ark/QA-data/ (Links to an external site.),https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs (Links to an external site.)

- Dataset has been splitted into 70/30 Train and Test (no validation set has been used)
- Model architecture 
- training log text 
- prediction on 10 samples picked from the test dataset

## Examples 

		id	qid1	qid2	question1	question2	is_duplicate
	0	0	1	2	What is the step by step guide to invest in sh...	What is the step by step guide to invest in sh...	0
	1	1	3	4	What is the story of Kohinoor (Koh-i-Noor) Dia...	What would happen if the Indian government sto...	0
	2	2	5	6	How can I increase the speed of my internet co...	How can Internet speed be increased by hacking...	0
	3	3	7	8	Why am I mentally very lonely? How can I solve...	Find the remainder when [math]23^{24}[/math] i...	0
	4	4	9	10	Which one dissolve in water quikly sugar, salt...	Which fish would survive in salt water?	0
## Hyperparameters

	INPUT_DIM = len(SRC.vocab)
	OUTPUT_DIM = len(TRG.vocab)
	ENC_EMB_DIM = 256
	DEC_EMB_DIM = 256
	HID_DIM = 512
	N_LAYERS = 2
	ENC_DROPOUT = 0.5
	DEC_DROPOUT = 0.5


## Model architecture

 

	Seq2Seq(
	  (encoder): Encoder(
	    (embedding): Embedding(14573, 256)
	    (rnn): LSTM(256, 512, num_layers=2, dropout=0.5)
	    (dropout): Dropout(p=0.5, inplace=False)
	  )
	  (decoder): Decoder(
	    (embedding): Embedding(14573, 256)
	    (rnn): LSTM(256, 512, num_layers=2, dropout=0.5)
	    (fc_out): Linear(in_features=512, out_features=14573, bias=True)
	    (dropout): Dropout(p=0.5, inplace=False)
	  )
	  )

## logs

	Epoch: 01 | Time: 5m 11s
		Train Loss: 4.789 | Train PPL: 120.212
		 Val. Loss: 4.370 |  Val. PPL:  79.053
	Epoch: 02 | Time: 5m 11s
		Train Loss: 3.653 | Train PPL:  38.598
		 Val. Loss: 3.803 |  Val. PPL:  44.848
	Epoch: 03 | Time: 5m 11s
		Train Loss: 3.018 | Train PPL:  20.452
		 Val. Loss: 3.044 |  Val. PPL:  20.990
	Epoch: 04 | Time: 5m 10s
		Train Loss: 2.400 | Train PPL:  11.021
		 Val. Loss: 2.390 |  Val. PPL:  10.908
	Epoch: 05 | Time: 5m 13s
		Train Loss: 1.883 | Train PPL:   6.576
		 Val. Loss: 1.894 |  Val. PPL:   6.644
	Epoch: 06 | Time: 5m 12s
		Train Loss: 1.491 | Train PPL:   4.440
		 Val. Loss: 1.553 |  Val. PPL:   4.724
	Epoch: 07 | Time: 5m 11s
		Train Loss: 1.197 | Train PPL:   3.309
		 Val. Loss: 1.306 |  Val. PPL:   3.690
	Epoch: 08 | Time: 5m 12s
		Train Loss: 0.973 | Train PPL:   2.645
		 Val. Loss: 1.146 |  Val. PPL:   3.145
	Epoch: 09 | Time: 5m 13s
		Train Loss: 0.804 | Train PPL:   2.233
		 Val. Loss: 1.011 |  Val. PPL:   2.748
	Epoch: 10 | Time: 5m 12s
		Train Loss: 0.673 | Train PPL:   1.960
		 Val. Loss: 0.924 |  Val. PPL:   2.520
	Epoch: 11 | Time: 5m 12s
		Train Loss: 0.573 | Train PPL:   1.773
		 Val. Loss: 0.863 |  Val. PPL:   2.370
	Epoch: 12 | Time: 5m 10s
		Train Loss: 0.494 | Train PPL:   1.639
		 Val. Loss: 0.810 |  Val. PPL:   2.248
	Epoch: 13 | Time: 5m 13s
		Train Loss: 0.431 | Train PPL:   1.539
		 Val. Loss: 0.775 |  Val. PPL:   2.170
	Epoch: 14 | Time: 5m 12s
		Train Loss: 0.381 | Train PPL:   1.464
		 Val. Loss: 0.735 |  Val. PPL:   2.086
	Epoch: 15 | Time: 5m 13s
		Train Loss: 0.338 | Train PPL:   1.403
		 Val. Loss: 0.712 |  Val. PPL:   2.039

## Output

![image](https://user-images.githubusercontent.com/16242779/123306733-490a4d80-d53f-11eb-8def-1a0821db340b.png)


## Assignment details on quiz section:
Once done, please upload the file to GitHub and proceed to answer these questions in the S7 - Assignment Solutions, where these questions are asked:
Share the link to your GitHub repo (100 pts for code quality/file structure/model accuracy) (100 pts)
Share the link to your readme file (100 points for proper readme file), this file can be the second part of your Part 1 Readme (basically you can have only 1 Readme, describing both assignments if you want) (100 pts)
Copy-paste the code related to your dataset preparation for both datasets.  (100 pts)
