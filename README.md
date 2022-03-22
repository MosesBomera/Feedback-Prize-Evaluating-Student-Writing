# [FeedbackPrizeEvaluatingStudentWriting](https://www.kaggle.com/c/feedback-prize-2021)
Writing is a critical skill for success. However, less than a third of high school seniors are proficient writers, according to the National Assessment of Educational Progress. Unfortunately, low-income, Black, and Hispanic students fare even worse, with less than 15 percent demonstrating writing proficiency. One way to help students improve their writing is via automated feedback tools, which evaluate student writing and provide personalized feedback.

In this challenge, the aim is to build an automated model to analyze argumentative writing elements from students grade 6-12. I used the [LeftmostSeg Model](https://arxiv.org/abs/2104.07217) and extended it from segmenting sentences to segmenting entire paragraphs and or groups of sentences.

### Improvements
- [ ]  Implement a more condensed vocabulary accounting for frequency of given words in the corpus.
- [ ]  Implement the [CharCNN](https://arxiv.org/abs/1509.01626) layer.
- [ ]  Use a transformer layer (Bert, and or Roberta).
