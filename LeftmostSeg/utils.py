import time
from itertools import chain
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .misc import iob_tagging
from .misc import recursionize
from .misc import f1_scoring
from .eval import scoreFeedbackComp


class LexicalAlphabet(object):
    PAD_SYM, UNK_SYM = "[PAD]", "[UNK]"

    def __init__(self):
        super(LexicalAlphabet, self).__init__()
        self._idx_to_item = []
        self._item_to_idx = {}
        self.add(LexicalAlphabet.PAD_SYM)
        self.add(LexicalAlphabet.UNK_SYM)

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def index(self, item):
        try:
            return self._item_to_idx[item]
        except KeyError:
            return self._item_to_idx[self.UNK_SYM]

    def __len__(self):
        return len(self._idx_to_item)


class LabelAlphabet(object):
    def __init__(self):
        super(LabelAlphabet, self).__init__()
        self._idx_to_item = []
        self._item_to_idx = {}

    def add(self, item):
        if item not in self._item_to_idx:
            self._item_to_idx[item] = len(self._idx_to_item)
            self._idx_to_item.append(item)

    def get(self, idx):
        return self._idx_to_item[idx]

    def index(self, item):
        return self._item_to_idx[item]

    def __len__(self):
        return len(self._idx_to_item)


class FeedbackPrizeDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.essayIds = df.id.unique()
        self.essayPaths = df[
            ['id', 'text_path']
        ].drop_duplicates(['id', 'text_path']).reset_index(drop=True)
        
    def __getitem__(self, idx):
        # Get the essay ID.
        essayId = self.essayIds[idx]
        # Get the essay file path.
        essayFilePath = self.essayPaths[self.essayPaths.id==essayId].iloc[0]['text_path']
        # Get the essay text.
        with open(essayFilePath) as f:
            # Ensure lowercaps.
            essayText = f.read().lower().split()
        # Get the text segments & labels.
        essaySegmentsBounds = self.df[self.df.id==essayId].segment_bounds.values
        segmentLabels = self.df[self.df.id==essayId].discourse_type.values
        labels = getFullLabelSequence(essaySegmentsBounds, segmentLabels, len(essayText))
        return [essayId, essayText, labels]
    
    def __len__(self):
        return len(self.essayIds)


def calculateWordIndices(full_text, discourse_start, discourse_end):
    """https://www.kaggle.com/kenkrige/predictionstring-100-consistent"""
    start_index = len(full_text[:discourse_start].split())
    token_len = len(full_text[discourse_start:discourse_end].split())
    output = list(range(start_index, start_index + token_len))
    if output[-1] >= len(full_text.split()):
        output = list(range(start_index, start_index + token_len-1))
    return " ".join(str(x) for x in output)


def getSegmentBounds(string):
    """Get the starting index and last index of a given segment."""
    string = string.split()
    return int(string[0]), int(string[-1])


def getFullLabelSequence(segmentSequence,labelSequence, textArrayLen):
    """Get the full label sequence for a given essay."""
    def getGaps(start, end, segmentSequence):
        """Get gaps in a sequence. https://stackoverflow.com/a/63814623/16419190"""
        ranges = sorted(segmentSequence)
        gaps = chain((start - 1,), chain.from_iterable(segmentSequence), (end + 1,))
        return [(x+1, y-1) for x, y in zip(gaps, gaps) if x+1 < y]
    
    # Sort the segment sequence. (Might be redundant.)
    segmentSequence = sorted(segmentSequence)
    # Get gaps.
    gaps = getGaps(0, textArrayLen - 1, segmentSequence)
    # Add discourse types to the segmentSequence.
    segmentSequence = [ss + (ls,) for ss, ls in zip(segmentSequence, labelSequence)]
    gapSequence = [(segment[0], segment[1], 'Untyped') for segment in gaps]
    # Sort the sequence by the second element.
    segmentSequence = sorted(segmentSequence + gapSequence, key=lambda seg: seg[1])
    for idx in range(len(segmentSequence)):
        try:
            if segmentSequence[idx+1][0] == segmentSequence[idx][1]:
                # Shift the next bound by one from the end bound of the previous bound.
                segmentSequence[idx+1] = (
                    segmentSequence[idx][1] + 1,segmentSequence[idx+1][1], segmentSequence[idx+1][2]
                    )
        except IndexError:
            continue
    return segmentSequence


def getPredictionString(ids, labels, classColumn):
    """Convert the model segmentation label to the competiton label."""
    if not isinstance(ids, list) and not isinstance(labels[0], list):
        ids, labels = [ids], [labels]
    labelsDf = pd.DataFrame()
    for id_, labels_ in zip(ids, labels):
        for label in labels_:
            # Ignore 'Untyped' class.
            if label[2] == 'Untyped':
                continue
            labelsDf = labelsDf.append(
                {
                    'id': id_,
                    classColumn: label[2],
                    'predictionstring': ' '.join(str(x) for x in list(range(label[0], label[1] + 1)))
                }, ignore_index=True
            )
    return labelsDf


def setUpVocab(df: "Trainset dataframe"):
    """Set up the lexical and label vocabulary.
    Introduce a new type 'Untyped' for the unlabeled gaps in the essays.
    """
    lexicalVocab = LexicalAlphabet()
    labelVocab = LabelAlphabet()
    # Load everything
    for txtPath in tqdm(df['text_path'].unique()):
        with open(txtPath) as f:
            text = f.read().lower().split()
            recursionize(lexicalVocab.add, text)
    # Labels vocabulary.
    recursionize(labelVocab.add, list(df.discourse_type.unique()) + ['Untyped'])
    return lexicalVocab, labelVocab


def getDataLoader(df, batchSize, ifShuffle):
    dataset = FeedbackPrizeDataset(df)
    return DataLoader(dataset, batch_size=batchSize, shuffle=ifShuffle, collate_fn=lambda x: list(zip(*x)))


class Procedure(object):
    @staticmethod
    def train(model, dataset, optimizer):
        model.train()
        time_start, total_loss = time.time(), 0.0

        for batch in tqdm(dataset, ncols=50):
            sentences, segments = batch[1], batch[2]
            penalty = model.estimate(sentences, segments)
            total_loss += penalty.item()

            optimizer.zero_grad()
            penalty.backward()
            optimizer.step()

        time_pass = time.time() - time_start
        return total_loss, time_pass

    @staticmethod
    def evaluate(model, dataset):
        model.eval()
        time_start = time.time()
        gtDf, predDf = pd.DataFrame(), pd.DataFrame()
        
        for batch in tqdm(dataset, ncols=50):
            essay_ids = batch[0]
            seqs, segments = batch[1], batch[2]
            with torch.no_grad():
                predictions = model.predict(seqs)

            # Get the prediction strings.
            gtDf = gtDf.append(getPredictionString(essay_ids, segments, 'discourse_type'), ignore_index=True)
            predDf = predDf.append(getPredictionString(essay_ids, predictions, 'class'), ignore_index=True)

        competition_score = scoreFeedbackComp(predDf, gtDf)
        return competition_score, time.time() - time_start
