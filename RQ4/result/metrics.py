import nltk
from nltk.translate.bleu_score import *
from nltk.translate.bleu_score import corpus_bleu
import collections
import os

is_local = True
if is_local:
    y_txt = 'golds.txt'
    hyp_txt = 'gens.txt'
    output_txt = None
else:
    y_txt = ''
    hyp_txt = ''
    output_txt = ''

#BLEU_M2(method2), BLEU_M3(method3), BLEU_DC(method4)
def bleu_m(refs,preds,smoothing_function):
    total_score = 0.0
    total_bleu1 = 0.0
    total_bleu2 = 0.0
    total_bleu3 = 0.0
    total_bleu4 = 0.0
    count = 0
    for ref,pred in zip(refs,preds):
        if len(pred.split()) > 4:
            score = nltk.translate.bleu([ref.split()],pred.split(),smoothing_function=smoothing_function)
            Cumulate_1_gram = nltk.translate.bleu([ref.split()], pred.split(), smoothing_function=smoothing_function, weights=(1, 0, 0, 0))
            Cumulate_2_gram = nltk.translate.bleu([ref.split()],pred.split(), smoothing_function=smoothing_function, weights=(0.5, 0.5, 0, 0))
            Cumulate_3_gram = nltk.translate.bleu([ref.split()],pred.split(), smoothing_function=smoothing_function, weights=(0.33, 0.33, 0.33, 0))
            Cumulate_4_gram = nltk.translate.bleu([ref.split()],pred.split(), smoothing_function=smoothing_function, weights=(0.25, 0.25, 0.25, 0.25))
            total_score += score
            total_bleu1 += Cumulate_1_gram
            total_bleu2 += Cumulate_2_gram
            total_bleu3 += Cumulate_3_gram
            total_bleu4 += Cumulate_4_gram
            count += 1
    return total_score * 100 / count, total_bleu1 * 100 / count, total_bleu2 * 100 / count, \
           total_bleu3 * 100 / count, total_bleu4 * 100 / count

#ROUGE-L
def myrouge(refs,pred,beta=1e2,eps=1e-2):
    max_score = 0.
    for ref in refs:
        dp = []
        for i in range(len(ref)):
            dp.append([])
            for j in range(len(pred)):
                dp[i].append(0)
                if i==0 or j==0:
                    if ref[i]==pred[j]:
                        dp[i][j] = 1
                    if i>0: dp[i][j] = max([dp[i][j],dp[i-1][j]])
                    elif j>0: dp[i][j] = max([dp[i][j],dp[i][j-1]])
        for i in range(1,len(ref)):
            for j in range(1,len(pred)):
                dp[i][j] = max([dp[i][j-1],dp[i-1][j]])
                if pred[j]==ref[i]:
                    dp[i][j] = max([dp[i][j],dp[i-1][j-1]+1])
        lcs = max([eps,1.*dp[len(ref)-1][len(pred)-1]])
        rec = lcs / len(ref)
        pre = lcs / len(pred)
        score = (1. + beta**2) * rec * pre / (rec + beta**2 * pre)
        max_score = max([score,max_score])
    return max_score

def ROUGE_L(refs,preds):
    rouge_l = 0.0
    for ref, pred in zip(refs, preds):
        if len(pred.split()) > 4:
            rouge_l += myrouge([ref.split()], pred.split())
    return rouge_l / len(refs)

def count_ngrams(words, n=4):
    counts = {}
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] = counts.get(ngram, 0)+1
    return counts

def cook_refs(refs, n=4):

    refs = [ref for ref in refs]
    maxcounts = {}
    for ref in refs:
        counts = count_ngrams(ref, n)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
    return ([len(ref) for ref in refs], maxcounts)

eff_ref_len = "shortest"
def cook_test(test, reflens, refmaxcounts, n=4):
    # test = normalize(test)
    result = {}
    result["testlen"] = len(test)

    # Calculate effective reference sentence length.

    if eff_ref_len == "shortest":
        result["reflen"] = min(reflens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(reflens)) / len(reflens)
    elif eff_ref_len == "closest":
        min_diff = None
        for reflen in reflens:
            if min_diff is None or abs(reflen - len(test)) < min_diff:
                min_diff = abs(reflen - len(test))
                result['reflen'] = reflen

    result["guess"] = [max(len(test) - k + 1, 0) for k in range(1, n + 1)]

    result['correct'] = [0] * n
    counts = count_ngrams(test, n)
    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result

def score_cooked(allcomps, n=4, ground=0, smooth=1):
    totalcomps = {'testlen':0, 'reflen':0, 'guess':[0]*n, 'correct':[0]*n}
    for comps in allcomps:
        for key in ['testlen','reflen']:
            totalcomps[key] += comps[key]
        for key in ['guess','correct']:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]
    logbleu = 0.0
    all_bleus = []
    for k in range(n):
      correct = totalcomps['correct'][k]
      guess = totalcomps['guess'][k]
      addsmooth = 0
      if smooth == 1 and k > 0:
        addsmooth = 1
      logbleu += math.log(correct + addsmooth + sys.float_info.min)-math.log(guess + addsmooth)
      if guess == 0:
        all_bleus.append(-10000000)
      else:
        all_bleus.append(math.log(correct + sys.float_info.min)-math.log( guess ))

    logbleu /= float(n)
    all_bleus.insert(0, logbleu)

    brevPenalty = min(0,1-float(totalcomps['reflen'] + 1)/(totalcomps['testlen'] + 1))
    for i in range(len(all_bleus)):
      if i ==0:
        all_bleus[i] += brevPenalty
      all_bleus[i] = math.exp(all_bleus[i])
    return all_bleus

#METEOR，NLTK>=3.5
def mymeteor(reference,hypothesis):
    total_score = 0.0
    count = 0
    for ref, pred in zip(reference, hypothesis):
        ref = ref.split(' ')
        pred = pred.split(' ')
        score = nltk.translate.meteor_score.single_meteor_score(ref, pred)
        total_score += score
        count += 1
    avg_score = total_score / count
    return avg_score

def bleu_so_far(ref, pred):
    Ba = corpus_bleu(ref, pred)
    B1 = corpus_bleu(ref, pred, weights=(1,0,0,0))
    B2 = corpus_bleu(ref, pred, weights=(0,1,0,0))
    B3 = corpus_bleu(ref, pred, weights=(0,0,1,0))
    B4 = corpus_bleu(ref, pred, weights=(0,0,0,1))
    return Ba*100, B1*100,B2*100,B3*100,B4*100

def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)

def print_result(refs,preds):

    print("=======================================================================")
    print("                                 BLEU")
    print("=======================================================================")
    print("             |   BLEU-avg  |  BLEU1  |   BLEU2  |   BLEU3  |   BLEU4  |")
    print("-----------------------------------------------------------------------")
    print("BLEU_M2      |    %5.2f    |  %5.2f  |   %5.2f  |   %5.2f  |   %5.2f  |"%(bleu_m(refs,preds,SmoothingFunction().method2)))
    print("-----------------------------------------------------------------------")
    print("BLEU_M3      |    %5.2f    |  %5.2f  |   %5.2f  |   %5.2f  |   %5.2f  |"%(bleu_m(refs,preds,SmoothingFunction().method3)))
    print("-----------------------------------------------------------------------")
    print("BLEU_DC      |    %5.2f    |  %5.2f  |   %5.2f  |   %5.2f  |   %5.2f  |"%(bleu_m(refs,preds,SmoothingFunction().method4)))
    print("-----------------------------------------------------------------------")
    print("\n=======================================================================")
    print("                                 OTHER")
    print("=======================================================================")
    print("method       |   SCORE    |")
    print("-----------------------------------------------------------------------")
    print("ROUGE-L      |   %.2f    |"%(ROUGE_L(refs,preds)*100))
    print("-----------------------------------------------------------------------")


def evaluate2(reference, predictions, output_path):

    hypotheses = []
    with open(predictions, 'r') as file:
        for line in file:
            hypotheses.append(line.strip())

    references = []
    with open(reference, 'r') as file:
        for line in file:
            references.append(line.strip())

    print_result(references, hypotheses)
    print("METEOR       |   %.2f    |" % (mymeteor(references, hypotheses) * 100))


if __name__ == '__main__':
    evaluate2(y_txt, hyp_txt, output_txt)
