"""Build an evidence-based paper using the supplied strict OOXML template.

Run with the project .venv Python. No model training or inference is performed.
"""
from pathlib import Path
import copy
import csv
import hashlib
import json
import re
import zipfile
from xml.dom import minidom
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TEMPLATE = Path(r'C:\Users\oshan\Downloads\conference-template-a4.docx')
TITLE = 'Page Type Aware Visual Phishing Detection Using EfficientNetV2B0'
AUTHOR = 'M.A Oshan Rashmika'
W = 'http://purl.oclc.org/ooxml/wordprocessingml/main'
R = 'http://purl.oclc.org/ooxml/officeDocument/relationships'
A = 'http://purl.oclc.org/ooxml/drawingml/main'
WP = 'http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing'
PIC = 'http://purl.oclc.org/ooxml/drawingml/picture'

FILES = {
    'm1_training': 'model 1 login pp2/reports/15_efficientnetv2b0_top_viewport_training/training_configuration.json',
    'm1_selection': 'model 1 login pp2/reports/17_efficientnetv2b0_continued_finetuning/continuation_comparison.json',
    'm1_selected_test': 'model 1 login pp2/reports/18_efficientnetv2b0_diagnostic_test/diagnostic_test_metrics.json',
    'baseline': 'research_suite/results/baseline_20261011T084658Z_aa75e1/report.json',
    'm2_test': 'new model 2 pp2/reports/model2_effnetv2b0/final_test/final_test_results.json',
    'm2_external': 'new model 2 pp2/reports/model2_effnetv2b0/external_evaluation/phish_iris/evaluation/phish_iris_external_results.json',
    'same_brand': 'research_suite/results/same-brand_20261011T084659Z_ed55d1/report.json',
    'masking': 'research_suite/results/masking_20261011T085645Z_43da93/report.json',
    'verification': 'research_suite/verification.json',
    'registry': 'research_suite/models.json',
}
data = {k: json.loads((ROOT / v).read_text(encoding='utf-8-sig')) for k, v in FILES.items()}
checks = []

def verify_metrics(name, metrics, keys):
    tn, fp, fn, tp = (metrics[k] for k in keys)
    values = {'accuracy': (tn+tp)/(tn+fp+fn+tp), 'precision': tp/(tp+fp),
              'recall': tp/(tp+fn), 'fpr': fp/(fp+tn), 'f1': 2*tp/(2*tp+fp+fn)}
    aliases = {'recall': ['recall', 'recall_sensitivity'], 'fpr': ['false_positive_rate'], 'f1': ['f1','f1_score']}
    for metric, value in values.items():
        key = next(k for k in aliases.get(metric, [metric]) if k in metrics)
        assert abs(value - metrics[key]) < 1e-10, (name, key, value, metrics[key])
    checks.append({'check': name+' confusion-count metric consistency', 'passed': True,
                   'images': tn+fp+fn+tp, 'counts': [tn,fp,fn,tp]})

for key in ['m2_test', 'm2_external']:
    m = data[key]['locked_threshold_metrics']
    verify_metrics(key, m, ['true_negative','false_positive','false_negative','true_positive'])
    folder = Path(FILES[key]).parent
    csv_name = 'final_test_predictions.csv' if key == 'm2_test' else 'phish_iris_external_predictions.csv'
    rows = list(csv.DictReader((ROOT / folder / csv_name).open(encoding='utf-8-sig', newline='')))
    counts = [0,0,0,0]
    for row in rows:
        y, p = int(row['true_numeric_label']), int(row['predicted_numeric_label'])
        counts[y*2+p] += 1
        # Archived CSV scores are rounded, so labels are the authoritative count check.
    assert counts == [m[k] for k in ['true_negative','false_positive','false_negative','true_positive']]
    checks.append({'check': key+' archived prediction CSV agreement', 'passed': True, 'images': len(rows)})
for method in data['baseline']['methods']:
    verify_metrics(method['method'], method, ['tn','fp','fn','tp'])
verify_metrics('m1 selected checkpoint diagnostic', data['m1_selected_test'],
               ['true_negatives','false_positives','false_negatives','true_positives'])
verify_metrics('same-brand exploratory', data['same_brand']['metrics'][0], ['tn','fp','fn','tp'])

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 8})

def architecture():
    fig, ax = plt.subplots(figsize=(3.45,3.35))
    ax.set(xlim=(-.005,1.005), ylim=(-.015,1)); ax.axis('off')
    def box(x,y,w,h,text,color='#eef2f5'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',
                                   facecolor=color,edgecolor='#334155',linewidth=.65))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(a,b,label=None):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':.8,'color':'#334155'})
        if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.023,label,ha='center',fontsize=7)
    box(.27,.90,.46,.075,'Input URL')
    box(.15,.735,.70,.11,'Browser capture\nScreenshot and rendered HTML')
    box(.18,.56,.64,.11,'Page type router\nURL patterns and HTML inputs')
    arrow((.5,.90),(.5,.845)); arrow((.5,.735),(.5,.67))
    box(.02,.295,.44,.15,'Login detector\nTop crop / bottom pad\n384 by 160 pixels','#e8f0f8')
    box(.54,.295,.44,.15,'General detector\nAspect fit / center pad\n384 by 224 pixels','#edf3e9')
    arrow((.36,.56),(.24,.445)); arrow((.64,.56),(.76,.445))
    ax.text(.12,.50,'Login',ha='center',fontsize=7)
    ax.text(.87,.50,'General',ha='center',fontsize=7)
    box(.02,.13,.44,.10,'EfficientNetV2B0\nThreshold 0.466','#e8f0f8')
    box(.54,.13,.44,.10,'EfficientNetV2B0\nThreshold 0.392','#edf3e9')
    arrow((.24,.295),(.24,.23)); arrow((.76,.295),(.76,.23))
    box(.18,.008,.64,.063,'Label, score and routing metadata')
    arrow((.24,.13),(.36,.071)); arrow((.76,.13),(.64,.071))
    fig.savefig(OUT/'architecture.png',dpi=350,bbox_inches='tight',pad_inches=.02)
    fig.savefig(OUT/'architecture.pdf',bbox_inches='tight',pad_inches=.02)
    plt.close(fig)

def confusion():
    fig, axes = plt.subplots(1,2,figsize=(3.45,1.85))
    for ax,key,title in zip(axes,['m2_test','m2_external'],['Fixed test','External (preliminary)']):
        m = data[key]['locked_threshold_metrics']
        counts = [[m['true_negative'],m['false_positive']],[m['false_negative'],m['true_positive']]]
        # Use counts and an identical scale so the panels remain comparable.
        ax.imshow(counts,cmap='Blues',vmin=0,vmax=1500)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(counts[i][j]),ha='center',va='center',fontsize=8,
                        color='white' if counts[i][j]>750 else '#111827')
        ax.set_xticks([0,1],['Legit.','Phish.'],fontsize=7)
        ax.set_yticks([0,1],['Legit.','Phish.'],fontsize=7)
        ax.set_title(title,fontsize=8)
        ax.set_xlabel('Predicted class',fontsize=7)
    axes[0].set_ylabel('True class',fontsize=7)
    fig.tight_layout(pad=.4)
    fig.savefig(OUT/'model2_confusion.png',dpi=350,bbox_inches='tight',pad_inches=.02)
    fig.savefig(OUT/'model2_confusion.pdf',bbox_inches='tight',pad_inches=.02)
    plt.close(fig)

architecture(); confusion()

CONTENT = []
def add(kind, text, extra=None): CONTENT.append((kind,text,extra))

add('abstract', 'Abstract—Visual phishing detection analyzes the appearance of webpages, but evaluation can be distorted by screenshot collection artifacts and differences between datasets. This paper presents a page type aware component that routes login pages and general webpages to separate EfficientNetV2B0 classifiers. The models use transfer learning, distinct image preparation pipelines, and validation selected decision thresholds. Saved evaluation artifacts show that the general webpage classifier achieves 91.36 percent accuracy, 95.66 percent phishing recall, and an F1 score of 0.9102 on 2765 fixed test images. A preliminary external evaluation on 2449 internally deduplicated Phish Iris images achieves 81.91 percent accuracy and an F1 score of 0.8187, with cross dataset overlap checking incomplete. The deployed login classifier achieves 81.54 percent diagnostic accuracy on 428 images, whereas a classifier using only original image dimensions achieves 100 percent accuracy on the same split. This finding exposes a strong dataset shortcut and limits claims about learned visual cues. The contribution is an implemented detection component and an evidence based assessment of its performance, checkpoint selection, and remaining validation needs. End to end routing performance and unseen brand generalization remain unestablished.')
add('keywords', 'Keywords—visual phishing detection, EfficientNetV2B0, transfer learning, page type routing, dataset bias, cross dataset evaluation')
add('h1','Introduction')
add('p','Phishing webpages can imitate familiar interfaces to persuade visitors to disclose credentials. Screenshot analysis provides a way to study the visual presentation of a page, including forms, text arrangement, colors, and branding. Earlier visual approaches include Phishpedia, which identifies brand impersonation through logo recognition and matching [1]. Visual similarity also creates an evaluation difficulty: a detector may distinguish the collection procedures of two classes instead of identifying evidence of phishing.')
add('p','Login interfaces and general webpages have different visual structures. A login screenshot often centers on a small credential form, while a general webpage can include navigation, illustrations, and multiple content regions. This motivates investigating separate image preparation and classification paths. However, architectural motivation alone does not establish that routing improves detection. A comparison against a single model on the same routed evaluation set is required to demonstrate that benefit.')
add('p','This work examines the implemented Micro Visual Cue Component. Its scope is binary screenshot classification, with URL and hypertext markup language (HTML) evidence used to select the classifier. It does not infer a target brand or compare a page domain with an authorized brand domain. The name describes the project component; the current classifiers process whole prepared images and do not explicitly extract a predefined set of micro visual features.')
add('p','The study has three contributions: an implemented pipeline connecting webpage capture, deterministic page type routing, two transfer learning classifiers, and an application programming interface (API); a reconstruction of training and evaluation protocols from saved manifests and reports; and a diagnostic analysis using simple baselines, preliminary external evaluation, same brand examples, and explanation tools. All numerical results below come from existing artifacts. No new training, threshold tuning, or live URL experiment was performed in preparing this paper.')
add('h1','Related Work')
add('p','Phishpedia combines logo detection and brand matching to identify visually impersonated webpages [1]. Its output includes a target brand, and its training approach does not require phishing examples. The present component instead trains binary classifiers on legitimate and phishing screenshots. These approaches address related but distinct decisions; their published metrics cannot be treated as directly comparable without shared data and a common evaluation protocol.')
add('p','EfficientNetV2 introduces convolutional architectures designed to improve training efficiency and parameter efficiency [2]. This work uses the B0 implementation as an ImageNet initialized feature extractor, followed by partial fine tuning. The implementation choice provides a concrete transfer learning backbone; the project has not established an inference speed or energy advantage over other architectures.')
add('p','Phish360 provides URL, HTML, and screenshot data for multimodal research [3]. Here, its screenshots support training and evaluating the general detector; the classifier does not jointly fuse those modalities. Phish-Iris supplies screenshots categorized into fourteen phishing brands and an additional legitimate category [4]. This study uses a local cleaned binary mapping of that collection for preliminary external evaluation. Counts reported here describe the local audited copies, rather than assuming that they exactly reproduce every sample in the published releases.')
add('p','Gradient weighted class activation mapping (Grad-CAM) uses gradients and convolutional activations to produce spatial explanations [5]. The research suite includes Grad-CAM and controlled white masking of specified regions. These tools are used as diagnostic probes of frozen classifiers. They do not replace independent evaluation or establish that a classifier relies on semantically correct features.')
add('h1','Component Design')
add('h2','Webpage Capture and Page Type Routing')
add('p','Fig. 1 summarizes the implementation. The capture routine uses Playwright with Chromium, a 1366 by 768 pixel viewport, a navigation timeout of 30 seconds, and a three second wait after document content loading. It stores the rendered HTML and a viewport screenshot rather than a full page screenshot. This rendering setup can differ from the collection setup used for training, so consistent browser capture is a relevant source of distribution shift.')
add('figure','architecture.png','Fig. 1. Implemented component. Routing chooses one classifier; the two model scores are not fused.')
add('p','The router selects the login path when it detects a recognized login or sign in URL pattern, or when the HTML contains a form together with password and user identifier inputs. Identifier evidence includes email input types, username or email autocomplete attributes, and relevant input attributes. A visible login navigation link alone does not trigger login routing. Pages without the required evidence use the general webpage path. The returned HIGH or MEDIUM confidence is a rule label, not a learned or calibrated probability.')
add('p','HTML processing uses regular expressions and approximate visible text extraction, so it is not a full document object model analysis. Inputs need not be associated with the same form in this heuristic. Passwordless authentication, controls inside frames or shadow trees, and pages with incomplete rendering can therefore challenge routing. The available classifier experiments select the model from manifests and do not measure router accuracy.')
add('h2','Model Specific Image Preparation')
add('p','For Model 1, the login detector, the image is converted to red green blue (RGB), with exchangeable image file orientation handled. Its width is resized to 384 pixels using Lanczos interpolation while preserving aspect ratio. The upper 160 pixels are retained if the resized height is sufficient; otherwise the image is aligned at the top of a white 384 by 160 canvas. This defines a consistent top viewport, but may remove a form positioned lower on a page.')
add('p','For Model 2, the general detector, the image is decoded with three channels. The scale is the smaller of 384 divided by original width and 224 divided by original height. Bilinear antialiased resizing preserves aspect ratio, and centered white padding produces a 384 by 224 input. Both paths supply floating point pixels in the range zero to 255 because preprocessing is included inside EfficientNetV2B0. An additional division by 255 is not applied.')
add('h2','Classifiers and API Integration')
add('p','Both models end in a sigmoid output representing the phishing class score. Model 1 uses average pooled backbone features, dropout of 0.35, a 128 unit swish dense layer with L2 regularization of 0.00001, dropout of 0.20, and a single output unit. Model 2 uses global average pooling, dropout of 0.30, and a single output unit. The login model has 6083409 parameters in the saved configuration. Model capacity alone is not evidence of effective cue learning.')
add('p','A page is labeled phishing when its selected score reaches the model threshold: 0.466 for Model 1 and 0.392 for Model 2. FastAPI exposes health and prediction endpoints. Models are loaded during startup, and a lock serializes predictions in the integration implementation. Responses include the label, score, selected model, threshold, and routing metadata. The sigmoid values are reported by the software as probabilities, but no dedicated calibration study is available.')
add('h1','Experimental Method')
add('h2','Datasets and Split Construction')
add('p','Table I lists the sample counts used in the archived experiments. The login data have balanced legitimate and phishing classes in each split. Their recorded seed is 20260826. The split reports show no cross split path or exact hash leakage, but the test set had already been inspected during development. Consequently, subsequent login results are diagnostic and must not be presented as performance on a fresh final test set. Full source attribution and brand coverage of this local login collection remain documentation gaps.')
add('table','TABLE I. LOCAL DATASET AND SPLIT COUNTS',[
    ['Collection / split','Legit.','Phish.','Total'],
    ['Login train','996','996','1992'],['Login validation','213','213','426'],['Login diagnostic','214','214','428'],
    ['Phish360 train','2434','2434','4868'],['Phish360 validation','608','608','1216'],['Phish360 fixed test','1499','1266','2765'],
    ['Phish-Iris external','1370','1079','2449']])
add('p','The local Phish360 audit starts from 10734 valid images. Review and exact overlap cleaning exclude 19 training or validation images while preserving all 2765 fixed test images. Among 45 reviewed near duplicate candidate pairs, 17 are confirmed and 28 retained as false matches. Balanced group aware splitting with seed 42 yields 4868 training and 1216 validation images; 1866 surplus legitimate training or validation images are unused. The reports record no remaining exact cross split leakage or confirmed pair leakage. This does not exclude every possible shared template.')
add('p','The local external Phish-Iris audit finds 2849 valid images and removes 400 exact duplicate copies, leaving 2449 unique images. The original brand folders are mapped to phishing and the other folder to legitimate. Both original source partitions are included because this model was trained on a different collection. However, cross collection near duplicate checking against Phish360 is recorded as incomplete. The archived evaluation therefore remains preliminary, even though internal exact duplicates were removed.')
add('h2','Training and Threshold Selection')
add('p','Model 1 uses batch size 16 and Adam optimization. A frozen backbone phase permits up to 15 epochs at learning rate 0.001. Partial fine tuning permits up to 20 epochs at 0.00001, requesting the last 40 backbone layers while freezing batch normalization. Training only augmentation applies translation of 0.02, zoom between minus 0.03 and 0.03, and contrast of 0.08, with no horizontal flipping or rotation. Checkpoint selection uses validation average precision; the decision threshold is selected using validation Matthews correlation coefficient.')
add('p','A continuation experiment subsequently runs for 12 of a permitted 15 epochs at learning rate 0.0000025. It does not meet the recorded minimum validation average precision gain of 0.001, and its validation Matthews correlation coefficient decreases slightly. The selection report retains the earlier checkpoint with hash prefix 6be68e84. The component configuration nevertheless deploys the continuation checkpoint with prefix d4d112fc. This is a recorded mismatch between model selection and deployment. Results for these two checkpoints are reported separately, without inferring that continuation improved generalization.')
add('p','Model 2 uses batch size 16, a frozen backbone phase permitting 20 epochs at 0.001, and partial fine tuning permitting 15 epochs at 0.00001. Its augmentation includes translation of 0.02, zoom between minus 0.04 and 0.04, and contrast of 0.08. Validation precision recall area is monitored during training. Threshold 0.392 is chosen by maximum validation F1 and locked for the fixed test evaluation. The archived report states that the threshold was not changed after test inspection.')
add('h2','Metrics and Diagnostic Controls')
add('p','Phishing is the positive class. We report accuracy, phishing precision, recall, F1, and false positive rate (FPR). Recall equals true positives divided by all phishing images; FPR equals false positives divided by all legitimate images. Precision uses all positive predictions as its denominator, and F1 is the harmonic mean of precision and recall. Counts are included so that class imbalance and operational errors are visible.')
add('p','The login comparison adds two logistic controls fitted on the same training membership: a color and spatial layout representation using an 8 by 4 grid of RGB means plus eight bin channel histograms, and original width, height, and aspect ratio features. Standardization is fitted on training data only. Logistic thresholds use validation F1, whereas the convolutional neural network (CNN) uses its locked threshold. The controls are diagnostic and differ from CNN augmentation and model selection; they are not an exhaustive comparison of competing detectors.')
add('p','A twenty image same brand set contains four legitimate and sixteen phishing examples from Facebook, LinkedIn, Microsoft, and PayPal. Each brand has one legitimate and four phishing screenshots. Labels, canonical brands, and provenance require review. A saved audit against 2418 training and validation images reports six unresolved near duplicate candidates, so these examples cannot support an independent generalization claim. Controlled masking additionally probes two selected phishing images using equal area control regions on exact prepared tensors.')
add('h1','Results and Analysis')
add('h2','General Webpage Detection')
add('p','At the locked threshold, Model 2 correctly detects 1211 of 1266 phishing test images, misses 55, and falsely flags 184 of 1499 legitimate images. Its fixed test accuracy is 91.36 percent, recall is 95.66 percent, precision is 86.81 percent, and F1 is 0.9102. Receiver operating characteristic area under the curve is 0.9726. Table II reports the principal model results, and Fig. 2 shows the general detector confusion counts.')
add('table','TABLE II. ARCHIVED CLASSIFIER RESULTS',[
    ['Model / evidence','Acc. %','Rec. %','FPR %','F1'],
    ['M2 fixed test','91.36','95.66','12.27','0.9102'],
    ['M2 external*','81.91','92.68','26.57','0.8187'],
    ['M1 deployed diagnostic','81.54','77.10','14.02','0.8068'],
    ['M1 selected diagnostic','81.07','75.70','13.55','0.8000']])
add('note','*Preliminary external evidence; cross dataset overlap checks are incomplete. Login rows use different checkpoints on an already inspected diagnostic split. These rows are not a common benchmark comparison.')
add('figure','model2_confusion.png','Fig. 2. Model 2 confusion counts at threshold 0.392. External data are internally deduplicated but cross dataset independence is unverified.')
add('p','On the preliminary external data, Model 2 detects 1000 of 1079 phishing images and falsely flags 364 of 1370 legitimate images. Accuracy declines to 81.91 percent, precision to 73.31 percent, and F1 to 0.8187, while recall remains 92.68 percent. The accuracy decline is 9.45 percentage points and FPR increases by 14.29 percentage points. This pattern is consistent with sensitivity to dataset shift, but incomplete overlap checking and label provenance limit causal attribution and independence claims.')
add('p','The test archive also includes a default threshold of 0.5 as a reference. It reduces fixed test FPR to 9.01 percent while lowering recall to 93.36 percent. Its higher test F1 does not authorize replacing the validation selected threshold after observing the test results. A deployment threshold should be chosen on separate validation data using the intended costs of missed phishing pages and false alarms, followed by a new reserved evaluation.')
add('h2','Login Checkpoints and Dataset Shortcuts')
add('p','The deployed continuation checkpoint produces 184 true negatives, 30 false positives, 49 false negatives, and 165 true positives on the 428 image diagnostic set. Its accuracy is 81.54 percent and F1 is 0.8068. The earlier selected checkpoint produces 185 true negatives, 29 false positives, 52 false negatives, and 162 true positives, with 81.07 percent accuracy. The modest numerical differences do not establish a reliable improvement, particularly because the split is already inspected and the continuation checkpoint failed its validation selection gates.')
add('p','Table III gives the diagnostic control comparison for the deployed checkpoint. The color and layout control reaches 66.36 percent accuracy and 49.53 percent FPR. More significantly, original dimensions alone produce perfect classification: 214 true negatives and 214 true positives, with no errors. This establishes that original image geometry is highly predictive of class in these splits. It does not establish that the CNN directly accesses original dimensions, because its inputs are prepared to fixed sizes. Resizing, cropping, padding, and source specific visual artifacts nevertheless remain plausible shortcut channels requiring controlled testing.')
add('table','TABLE III. LOGIN DIAGNOSTIC CONTROLS',[
    ['Method','Acc. %','Rec. %','FPR %','F1'],
    ['Color / layout logistic','66.36','82.24','49.53','0.7097'],
    ['Dimensions logistic','100.00','100.00','0.00','1.0000'],
    ['Deployed login CNN','81.54','77.10','14.02','0.8068']])
add('p','The dimensions result is an adverse validity finding rather than a deployment recommendation. A meaningful next experiment would collect both classes under matched browser, viewport, and rendering conditions, preserve source and template groups when splitting, and evaluate newly trained checkpoints on reserved data. Reusing these checkpoints with a newly proposed split would not make their original training brand disjoint.')
add('h2','Same Brand and Explanation Diagnostics')
add('p','The exploratory login set yields four true negatives, zero false positives, seven false negatives, and nine true positives. Accuracy is 65.00 percent, phishing recall is 56.25 percent, and F1 is 0.7200. Brand recalls are 75 percent for Facebook, 25 percent for LinkedIn, 50 percent for Microsoft, and 75 percent for PayPal. With only one legitimate screenshot per brand, zero observed false positives is weak evidence about the false alarm rate. The unresolved overlap candidates further restrict interpretation.')
add('p','The suite exports class heatmaps, prepared inputs, source images, scores, and checkpoint metadata. Its archived verification reports exact forward score agreement for the explanation graph and checks 44 class maps. This supports numerical consistency of the explanation implementation, not semantic correctness. Heatmaps are normalized per image, and their intensity should not be compared across examples.')
add('p','Masking the selected Model 2 form lowers its score from 0.9682 to 0.9461, while an equal area control lowers it to 0.9587. Masking its branding region instead increases the score to 0.9833; the branding control produces 0.9690. For the selected Model 1 image, form masking increases the score from 0.9646 to 0.9711, and both controls also increase it. All seven perturbations retain the phishing label. These two sample probes show heterogeneous responses and do not establish brand independence, form dependence, or an explanation performance rate.')
add('h1','Limitations and Future Evaluation')
add('p','The strongest available result concerns a standalone general classifier on the preserved local Phish360 test split. It does not measure the complete capture and routing pipeline. End to end validation needs labeled page types and phishing labels on the same reserved URL collection, including capture failures, rendering variability, and incorrect routing. A single detector and the routed system should then be compared on identical cases. No measured latency, throughput, or calibrated risk estimates are supplied here.')
add('p','For the login detector, dimensions predictability, incomplete collection provenance, and repeated diagnostic test use substantially limit scientific interpretation. For external evaluation, internal deduplication is insufficient until cross dataset exact and near duplicate checks are complete. A complete reviewed training brand inventory is also unavailable, so no unseen brand evaluation is established. Existing brand names or heatmaps cannot substitute for a held out brand protocol.')
add('p','Future work should reconcile the deployed login checkpoint with the recorded selection decision, document dataset licensing and collection sources, resolve overlap candidates, and create matched acquisition data. New training experiments should reserve domain, template, brand, and preferably time based holdouts before model selection. Evaluation should retain counts, per brand errors, uncertainty estimates appropriate to correlated templates, and checkpoint hashes. Region studies need multiple brands, both correct predictions and errors, and semantically reviewed controls fixed before scores are examined.')
add('h1','Conclusion')
add('p','The Micro Visual Cue Component implements page type routing and two EfficientNetV2B0 screenshot classifiers with distinct preparation pipelines and locked decision thresholds. Archived general detector results show high phishing recall on the fixed local test set, alongside a substantial false positive increase on preliminary external data. The login dimensions baseline reveals a major dataset shortcut, and the deployed checkpoint differs from the checkpoint retained by validation selection. These findings support an implementation and diagnostic study while identifying the conditions needed for stronger claims: matched screenshot acquisition, verified data independence, consistent checkpoint selection, and reserved end to end evaluation.')
add('refs','References')
REFERENCES = [
    '[1] Y. Lin et al., “Phishpedia: A hybrid deep learning based approach to visually identify phishing webpages,” in Proc. 30th USENIX Security Symposium, 2021, pp. 3793–3810. https://www.usenix.org/conference/usenixsecurity21/presentation/lin',
    '[2] M. Tan and Q. Le, “EfficientNetV2: Smaller models and faster training,” in Proc. 38th International Conference on Machine Learning, PMLR, vol. 139, 2021, pp. 10096–10106. https://proceedings.mlr.press/v139/tan21a.html',
    '[3] A. H. A. Almakhamreh and A. S. Bozkir, “CrossPhire: Benefiting multimodality for robust phishing web page identification,” Applied Sciences, vol. 16, no. 2, article 751, 2026, doi: 10.3390/app16020751. Dataset: https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/',
    '[4] A. S. Bozkir and F. C. Dalgic, “Phish-Iris dataset: A small scale multi-class phishing web page screenshots dataset,” Mendeley Data, version 1, 2018, doi: 10.17632/tk4fkswtj5.1. https://data.mendeley.com/datasets/tk4fkswtj5/1',
    '[5] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE International Conference on Computer Vision, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74. https://arxiv.org/abs/1610.02391',
]
for ref in REFERENCES: add('ref',ref)

md = ['# '+TITLE, '', AUTHOR, 'Sri Lanka Institute of Information Technology, Sri Lanka', 'Oshanrashmika180@gmail.com','']
for kind,text,extra in CONTENT:
    if kind in ['h1','refs']: md.extend(['## '+text,''])
    elif kind == 'h2': md.extend(['### '+text,''])
    elif kind == 'table':
        md.extend(['**'+text+'**','', '| '+' | '.join(extra[0])+' |', '| '+' | '.join(['---']*len(extra[0]))+' |'])
        md.extend('| '+' | '.join(row)+' |' for row in extra[1:]); md.append('')
    elif kind == 'figure': md.extend(['!['+extra+']('+text+')','',extra,''])
    else: md.extend([text,''])
(OUT/'research_paper.md').write_text('\n'.join(md),encoding='utf-8')

with zipfile.ZipFile(TEMPLATE) as z:
    parts = {name:z.read(name) for name in z.namelist()}
document = minidom.parseString(parts['word/document.xml'])
body = document.getElementsByTagNameNS(W,'body')[0]
sections = [s.cloneNode(True) for s in document.getElementsByTagNameNS(W,'sectPr')]
while body.firstChild: body.removeChild(body.firstChild)

def fragment(xml):
    xml = f'<root xmlns:w="{W}" xmlns:r="{R}" xmlns:a="{A}" xmlns:wp="{WP}" xmlns:pic="{PIC}">{xml}</root>'
    node = minidom.parseString(xml).documentElement.firstChild
    return document.importNode(node,True)

# Namespaces used by the embedded image markup must exist in the final document.
for prefix,namespace in [('a',A),('pic',PIC)]:
    document.documentElement.setAttribute('xmlns:'+prefix,namespace)

def paragraph(text,style='BodyText',size=None,center=False,keep=False):
    ppr = f'<w:pStyle w:val="{style}"/>'
    if keep: ppr += '<w:keepNext/>'
    if center: ppr += '<w:jc w:val="center"/>'
    # Body font is 10pt in the template sample's direct formatting.
    rpr = f'<w:sz w:val="{size or 20}"/><w:szCs w:val="{size or 20}"/>' if size or style in ['BodyText','Affiliation','Heading1','Heading2','Heading5','Keywords'] else ''
    p = fragment(f'<w:p><w:pPr>{ppr}</w:pPr><w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')
    body.appendChild(p)
    return p

paragraph(TITLE,'papertitle')
paragraph(AUTHOR,'Author')
paragraph('Sri Lanka Institute of Information Technology','Affiliation')
paragraph('Sri Lanka','Affiliation')
paragraph('Oshanrashmika180@gmail.com','Affiliation')
end_title = paragraph('','Affiliation',size=2)
title_sect = sections[0]
for node in list(title_sect.childNodes):
    if node.nodeType == node.ELEMENT_NODE and node.localName in ['footerReference','headerReference','titlePg']:
        title_sect.removeChild(node)
end_title.getElementsByTagNameNS(W,'pPr')[0].appendChild(title_sect)

rels = minidom.parseString(parts['word/_rels/document.xml.rels'])
relationship_ns = rels.documentElement.namespaceURI
image_index = 0

def image_paragraph(filename,caption):
    global image_index
    from PIL import Image
    image_index += 1
    with Image.open(OUT/filename) as im: width,height=im.size
    cx = int(3.37*914400); cy = round(cx*height/width)
    rid = 'rIdPaperImage'+str(image_index)
    target = 'media/paper_'+filename
    parts['word/'+target] = (OUT/filename).read_bytes()
    rel = rels.createElementNS(relationship_ns,'Relationship')
    for key,value in {'Id':rid,'Type':R+'/image','Target':target}.items(): rel.setAttribute(key,value)
    rels.documentElement.appendChild(rel)
    body.appendChild(fragment(f'''<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/><w:ind w:firstLine="0"/></w:pPr><w:r><w:drawing>
    <wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>
    <wp:docPr id="{image_index}" name="{filename}" descr="{escape(caption)}"/>
    <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
    <a:graphic><a:graphicData uri="{PIC}"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{filename}"/><pic:cNvPicPr/></pic:nvPicPr>
    <pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
    <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
    </pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''))
    paragraph(caption,'figurecaption')

def table(caption,rows):
    paragraph(caption,'tablehead',keep=True)
    # The exact two-column body geometry permits 243.3 points per column.
    widths = [2050] + [int((4866-2050)/(len(rows[0])-1))]*(len(rows[0])-1)
    xml = '<w:tbl><w:tblPr><w:tblW w:w="4866" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblBorders>'
    for border in ['top','left','bottom','right','insideH','insideV']:
        xml += f'<w:{border} w:val="single" w:sz="4" w:color="888888"/>'
    xml += '</w:tblBorders><w:tblCellMar><w:top w:w="45" w:type="dxa"/><w:left w:w="55" w:type="dxa"/><w:bottom w:w="45" w:type="dxa"/><w:right w:w="55" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'
    xml += ''.join(f'<w:gridCol w:w="{width}"/>' for width in widths)+'</w:tblGrid>'
    for i,row in enumerate(rows):
        xml += '<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if i==0 else '')+'</w:trPr>'
        for j,cell in enumerate(row):
            shade='<w:shd w:fill="EEF2F5"/>' if i==0 else ''
            style='tablecolhead' if i==0 else 'tablecopy'
            xml += f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>{shade}</w:tcPr><w:p><w:pPr><w:pStyle w:val="{style}"/><w:spacing w:after="0"/><w:ind w:firstLine="0"/><w:jc w:val="{ "start" if j==0 else "center" }"/></w:pPr><w:r><w:t>{escape(cell)}</w:t></w:r></w:p></w:tc>'
        xml += '</w:tr>'
    body.appendChild(fragment(xml+'</w:tbl>'))
    paragraph('','tablecopy',size=2)

for kind,text,extra in CONTENT:
    if kind=='table': table(text,extra)
    elif kind=='figure': image_paragraph(text,extra)
    else:
        style = {'abstract':'Abstract','keywords':'Keywords','h1':'Heading1','h2':'Heading2',
                 'refs':'Heading5','ref':'references','note':'tablefootnote'}.get(kind,'BodyText')
        paragraph(text,style)
body.appendChild(sections[3])
parts['word/document.xml'] = document.toxml(encoding='UTF-8')
parts['word/_rels/document.xml.rels'] = rels.toxml(encoding='UTF-8')

# Remove template guidance from all headers/footers; no conference identifiers were supplied.
for name in list(parts):
    if re.fullmatch(r'word/(header|footer)\d+\.xml',name):
        root_name = 'hdr' if '/header' in name else 'ftr'
        parts[name] = f'<?xml version="1.0" encoding="UTF-8"?><w:{root_name} xmlns:w="{W}"><w:p/></w:{root_name}>'.encode()
ct = minidom.parseString(parts['[Content_Types].xml'])
if not any(n.getAttribute('Extension')=='png' for n in ct.getElementsByTagName('Default')):
    node = ct.createElement('Default'); node.setAttribute('Extension','png'); node.setAttribute('ContentType','image/png'); ct.documentElement.appendChild(node)
parts['[Content_Types].xml'] = ct.toxml(encoding='UTF-8')
paper_path = OUT/'Micro_Visual_Cue_Research_Paper.docx'
with zipfile.ZipFile(paper_path,'w',zipfile.ZIP_DEFLATED) as z:
    for name,payload in parts.items(): z.writestr(name,payload)

# Validate output package, relationships, expected structure and factual evidence.
with zipfile.ZipFile(paper_path) as z:
    assert z.testzip() is None
    for name in z.namelist():
        if name.endswith('.xml') or name.endswith('.rels'): minidom.parseString(z.read(name))
    assert z.read('word/styles.xml') == parts['word/styles.xml']
    actual = minidom.parseString(z.read('word/document.xml'))
    texts = ' '.join(n.firstChild.data for n in actual.getElementsByTagNameNS(W,'t') if n.firstChild)
    for unwanted in ['Paper Title*','Identify applicable funding','This electronic document','G. Eason','IEEE conference templates contain guidance']:
        assert unwanted not in texts
    assert len(actual.getElementsByTagNameNS(W,'tbl')) == 3
    assert len(actual.getElementsByTagNameNS(W,'drawing')) == 2
    assert len(actual.getElementsByTagNameNS(W,'sectPr')) == 2
checks.append({'check':'DOCX XML, media, tables, sections and template guidance removal','passed':True})

provenance = {k:{'path':v,'sha256':hashlib.sha256((ROOT/v).read_bytes()).hexdigest()} for k,v in FILES.items()}
result = {'checks':checks,'sources':provenance,'template':str(TEMPLATE),
          'template_sha256':hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
          'paper_sha256':hashlib.sha256(paper_path.read_bytes()).hexdigest(),
          'visual_layout_rendered_in_word':False,
          'note':'Existing evidence audited; models were not retrained or re-evaluated. DOCX uses supplied template styles and strict OOXML.'}
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

notes = '''# Research paper review notes

The Word draft uses the supplied conference-template-a4.docx, its Times New Roman styles, A4 geometry, a full-width title/author block, and two-column body. The original template has not been changed. The paper includes three tables, two original figures, and five verified primary references. Heading numbering comes from the template. Figure PNG and vector PDF files are separate reusable assets.

Author: M.A Oshan Rashmika. Affiliation: Sri Lanka Institute of Information Technology, Sri Lanka. Email: Oshanrashmika180@gmail.com. No department or city was invented. No page limit was supplied.

## Items to resolve before submission

- The deployed login checkpoint differs from the checkpoint retained by validation selection. This discrepancy is disclosed in the paper; the implementation was not changed.
- The login test split has already been inspected and its dimensions-only baseline is perfectly predictive. Login metrics are diagnostic, not fresh final evaluation evidence.
- Phish-Iris results are preliminary. The archived evaluation ran despite a local cleaning note requiring overlap checking first; the result JSON explicitly records cross-dataset near-duplicate checking as incomplete. Complete exact and perceptual overlap checks before promoting these results to independent external evidence.
- Same-brand labels and brands need review, and six overlap candidates remain unresolved. Do not claim unseen-brand generalization.
- Document the original login collection, source URLs, acquisition procedure, dates, label provenance, and dataset permissions. Local counts differ from published dataset totals and are described as local audited copies.
- End-to-end routing accuracy, a matched single-model comparison, calibration, and latency are not available. They are not claimed.
- No research novelty or superiority over published systems is established by this draft. The contributions are scoped to the implemented component and its diagnostic evidence.
- Inspect the DOCX in Microsoft Word for final column flow, figure positioning, table wrapping, and conference-specific details before exporting a submission PDF. This build verifies document XML and package structure; it does not claim a Word rendering or final page count.

## Rebuild

From the module root: `.\\.venv\\Scripts\\python.exe research_paper\\build_paper.py`.

`research_paper.md` is an editable content copy. `build_paper.py` currently contains the content used to regenerate the DOCX. Existing training/evaluation artifacts and models are read only.

## Evidence map

'''
for key,path in FILES.items(): notes += f'- {key}: `{path}`\n'
notes += '''
Additional methods are supported by `Micro Visual Cue Component/src/visual_component.py`, `Micro Visual Cue Component/src/page_type_router.py`, `Micro Visual Cue Component/integration/visual_api.py`, both model training scripts, and the local split and cleaning summaries. `verification.json` records source hashes and recomputation of metrics from confusion counts, plus comparison against the archived Model 2 prediction CSVs.

## Verified external references

- Phishpedia: https://www.usenix.org/conference/usenixsecurity21/presentation/lin
- EfficientNetV2: https://proceedings.mlr.press/v139/tan21a.html
- Phish360 and CrossPhire citation: https://web.cs.hacettepe.edu.tr/~selman/phish360-dataset/
- Phish-Iris release: https://data.mendeley.com/datasets/tk4fkswtj5/1
- Grad-CAM: https://arxiv.org/abs/1610.02391
'''
(OUT/'REVIEW_NOTES.md').write_text(notes,encoding='utf-8')
print(json.dumps({'paper':str(paper_path),'words_approx':len(' '.join(md).split()),
                  'checks_passed':len(checks),'figures':2,'tables':3},indent=2))
